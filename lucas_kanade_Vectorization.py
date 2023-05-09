import cv2
import numpy as np
from scipy import signal
from scipy.interpolate import RectBivariateSpline, interp2d





# FILL IN YOUR ID
ID1 = 308345891
ID2 = 211670849

PYRAMID_FILTER = 1.0 / 256 * np.array([[1, 4, 6, 4, 1],
                                       [4, 16, 24, 16, 4],
                                       [6, 24, 36, 24, 6],
                                       [4, 16, 24, 16, 4],
                                       [1, 4, 6, 4, 1]])
X_DERIVATIVE_FILTER = np.array([[1, 0, -1],
                                [2, 0, -2],
                                [1, 0, -1]])
Y_DERIVATIVE_FILTER = X_DERIVATIVE_FILTER.copy().transpose()

WINDOW_SIZE = 5


def get_video_parameters(capture: cv2.VideoCapture) -> dict:
    """Get an OpenCV capture object and extract its parameters.
    Args:
        capture: cv2.VideoCapture object.
    Returns:
        parameters: dict. Video parameters extracted from the video.
    """
    fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    return {"fourcc": fourcc, "fps": fps, "height": height, "width": width,
            "frame_count": frame_count}


def build_pyramid(image: np.ndarray, num_levels: int) -> list[np.ndarray]:
    """Coverts image to a pyramid list of size num_levels.
    First, create a list with the original image in it. Then, iterate over the
    levels. In each level, convolve the PYRAMID_FILTER with the image from the
    previous level. Then, decimate the result using indexing: simply pick
    every second entry of the result.
    Hint: Use signal.convolve2d with boundary='symm' and mode='same'.
    Args:
        image: np.ndarray. Input image.
        num_levels: int. The number of blurring / decimation times.
    Returns:
        pyramid: list. A list of np.ndarray of images.
    Note that the list length should be num_levels + 1 as the in first entry of
    the pyramid is the original image.
    You are not allowed to use cv2 PyrDown here (or any other cv2 method).
    We use a slightly different decimation process from this function.
    """
    pyramid = [image.copy()]
    """INSERT YOUR CODE HERE."""
    for i in range(num_levels):
        img_lev = pyramid[i]
        h, w = img_lev.shape
        # Low-pass filter + decimation factor 2
        img_lev = signal.convolve2d(in1=img_lev, in2=PYRAMID_FILTER, mode='same', boundary='symm')
        img_lev = img_lev[0:h:2, 0:w:2]
        pyramid.append(img_lev)
    return pyramid

def my_warp_image(image: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    h, w = image.shape
    # Step 1: Compute scaling factors and resize u and v
    U_FACTOR = w / u.shape[1]
    V_FACTOR = h / v.shape[0]
    u = cv2.resize(u, (w, h), interpolation=cv2.INTER_LINEAR) * U_FACTOR
    v = cv2.resize(v, (w, h), interpolation=cv2.INTER_LINEAR) * V_FACTOR
    # Step 2: Compute new coordinates for pixels in image_warp
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    x_new = x + u
    y_new = y + v
    # Step 3: Interpolate pixel values using RectBivariateSpline
    interp = RectBivariateSpline(np.arange(h), np.arange(w), image)
    image_warp = interp.ev(y_new, x_new)
    # Step 4: Fill in NaN values with corresponding pixels from original image
    nan_mask = np.isnan(image_warp)
    image_warp[nan_mask] = image[nan_mask]
    return image_warp



def my_lucas_kanade_step(I1: np.ndarray,
                      I2: np.ndarray,
                      window_size: int) -> tuple[np.ndarray, np.ndarray]:
    # Initialize step
    du = np.zeros(I1.shape)
    dv = np.zeros(I1.shape)
    h, w = I1.shape
    epsilon = 1e-4
    # Step1:
    Ix = signal.convolve2d(in1=I2, in2=X_DERIVATIVE_FILTER, mode='same', boundary='symm')
    Iy = signal.convolve2d(in1=I2, in2=Y_DERIVATIVE_FILTER, mode='same', boundary='symm')
    # Step2:
    It = I2 - I1
    # Step3:
    Ixx = Ix * Ix
    Iyy = Iy * Iy
    Ixy = Ix * Iy
    Ixt = -1 * Ix * It
    Iyt = -1 * Iy * It

    # Compute sliding windows for the arrays
    window_shape = (window_size, window_size)
    Ixx_windows = np.lib.stride_tricks.sliding_window_view(Ixx, window_shape)
    Iyy_windows = np.lib.stride_tricks.sliding_window_view(Iyy, window_shape)
    Ixy_windows = np.lib.stride_tricks.sliding_window_view(Ixy, window_shape)
    Ixt_windows = np.lib.stride_tricks.sliding_window_view(Ixt, window_shape)
    Iyt_windows = np.lib.stride_tricks.sliding_window_view(Iyt, window_shape)

    # Calculate ATA and b using vectorized operations
    ATA_11 = np.sum(Ixx_windows, axis=(2, 3))
    ATA_12 = np.sum(Ixy_windows, axis=(2, 3))
    ATA_21 = ATA_12
    ATA_22 = np.sum(Iyy_windows, axis=(2, 3))
    ATA = np.stack([np.stack([ATA_11, ATA_12], axis=-1), np.stack([ATA_21, ATA_22], axis=-1)], axis=-2)
    det_ATA = np.linalg.det(ATA)
    b_1 = np.sum(Ixt_windows, axis=(2, 3))[:, :, np.newaxis]
    b_2 = np.sum(Iyt_windows, axis=(2, 3))[:, :, np.newaxis]
    b = np.concatenate([b_1, b_2], axis=-1)

    # Calculate U_V_LS using vectorized operations
    valid_indices = np.where(det_ATA > epsilon)
    U_V_LS = np.zeros((h, w, 2))
    U_V_LS[valid_indices[0] + window_size // 2, valid_indices[1] + window_size // 2] = np.linalg.solve(ATA[valid_indices], b[valid_indices])
    du, dv = U_V_LS[:, :, 0], U_V_LS[:, :, 1]
    return du, dv

def lucas_kanade_optical_flow(I1: np.ndarray,
                              I2: np.ndarray,
                              window_size: int,
                              max_iter: int,
                              num_levels: int) -> tuple[np.ndarray, np.ndarray]:
    """Calculate LK Optical Flow for max iterations in num-levels.
    Args:
        I1: np.ndarray. Image at time t.
        I2: np.ndarray. Image at time t+1.
        window_size: int. The window is of shape window_size X window_size.
        max_iter: int. Maximal number of LK-steps for each level of the pyramid.
        num_levels: int. Number of pyramid levels.
    Returns:
        (u, v): tuple of np.ndarray-s. Each one of the shape of the
        original image. v encodes the optical flow parameters in rows and u in
        columns.
    Recipe:
        (1) Since the image is going through a series of decimations,
        we would like to resize the image shape to:
        K * (2^(num_levels - 1)) X M * (2^(num_levels - 1)).
        Where: K is the ceil(h / (2^(num_levels - 1)),
        and M is ceil(h / (2^(num_levels - 1)).
        (2) Build pyramids for the two images.
        (3) Initialize u and v as all-zero matrices in the shape of I1.
        (4) For every level in the image pyramid (start from the smallest
        image):
          (4.1) Warp I2 from that level according to the current u and v.
          (4.2) Repeat for num_iterations:
            (4.2.1) Perform a Lucas Kanade Step with the I1 decimated image
            of the current pyramid level and the current I2_warp to get the
            new I2_warp.
          (4.3) For every level which is not the image's level, perform an
          image resize (using cv2.resize) to the next pyramid level resolution
          and scale u and v accordingly.
    """
    """INSERT YOUR CODE HERE.
        Replace image_warp with something else.
        """
    DOWN_FACTOR = 2
    h_factor = int(np.ceil(I1.shape[0] / (2 ** (num_levels - 1 + 1))))
    w_factor = int(np.ceil(I1.shape[1] / (2 ** (num_levels - 1 + 1))))
    IMAGE_SIZE = (w_factor * (2 ** (num_levels - 1 + 1)),
                  h_factor * (2 ** (num_levels - 1 + 1)))
    if I1.shape != IMAGE_SIZE:
        I1 = cv2.resize(I1, IMAGE_SIZE)
    if I2.shape != IMAGE_SIZE:
        I2 = cv2.resize(I2, IMAGE_SIZE)
    # create a pyramid from I1 and I2
    pyramid_I1 = build_pyramid(I1, num_levels)
    pyarmid_I2 = build_pyramid(I2, num_levels)
    # start from u and v in the size of smallest image
    u = np.zeros(pyarmid_I2[-1].shape)
    v = np.zeros(pyarmid_I2[-1].shape)
    """INSERT YOUR CODE HERE.Replace u and v with their true value."""
    for level in range(num_levels, -1, -1):
        I2_warp = my_warp_image(pyarmid_I2[level], u, v)
        for iter in range(max_iter):
            du, dv = my_lucas_kanade_step(I1=pyramid_I1[level], I2=I2_warp, window_size=window_size)
            u += du
            v += dv
            I2_warp = my_warp_image(pyarmid_I2[level], u, v)
        if level > 0:
            h_scale, w_scale = pyarmid_I2[level - 1].shape
            u = cv2.resize(u, (w_scale, h_scale)) * DOWN_FACTOR
            v = cv2.resize(v, (w_scale, h_scale)) * DOWN_FACTOR
    return u, v


def lucas_kanade_video_stabilization(input_video_path: str,
                                     output_video_path: str,
                                     window_size: int,
                                     max_iter: int,
                                     num_levels: int,
                                     start_rows: int = 10,
                                     start_cols: int = 2,
                                     end_rows: int = 30,
                                     end_cols: int = 30
                                     ) -> None:

    """INSERT YOUR CODE HERE."""
    cap = cv2.VideoCapture(input_video_path)
    params = get_video_parameters(cap)
    out = cv2.VideoWriter(output_video_path, fourcc=cv2.VideoWriter_fourcc(*'XVID'), fps=params["fps"],
                          frameSize=(params["width"], params["height"]), isColor=False)
    ret, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    out.write(gray_frame)

    h_factor = int(np.ceil(gray_frame.shape[0] / (2 ** (num_levels - 1 + 1))))
    w_factor = int(np.ceil(gray_frame.shape[1] / (2 ** (num_levels - 1 + 1))))
    IMAGE_SIZE = (w_factor * (2 ** (num_levels - 1 + 1)), h_factor * (2 ** (num_levels - 1 + 1)))
    gray_frame = cv2.resize(gray_frame, IMAGE_SIZE)
    u = np.zeros(gray_frame.shape, dtype=np.float)
    v = np.zeros(gray_frame.shape, dtype=np.float)
    prev_frame = gray_frame
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_frame = cv2.resize(gray_frame, IMAGE_SIZE)
            du, dv = lucas_kanade_optical_flow(I1=prev_frame, I2=gray_frame, window_size=window_size,
                                                      max_iter=max_iter, num_levels=num_levels)
            r_low_u, r_high_u = window_size // 2, du.shape[0] - window_size // 2
            c_low_u, c_high_u = window_size // 2, du.shape[1] - window_size // 2
            r_low_v, r_high_v = window_size // 2, dv.shape[0] - window_size // 2
            c_low_v, c_high_v = window_size // 2, dv.shape[1] - window_size // 2
            du_mean, dv_mean = np.mean(du[r_low_u:r_high_u, c_low_u:c_high_u]), np.mean(
                dv[r_low_v:r_high_v, c_low_v:c_high_v])
            # Part D
            u[r_low_u:r_high_u, c_low_u:c_high_u] += du_mean
            v[r_low_v:r_high_v, c_low_v:c_high_v] += dv_mean
            # Part E
            warp_frame = my_warp_image(gray_frame, u, v)
            warp_frame = warp_frame[start_rows:gray_frame.shape[0] - end_rows,
                         start_cols:gray_frame.shape[1] - end_cols]
            warp_frame = cv2.resize(warp_frame, (params["width"], params["height"]))
            out.write(warp_frame.astype('uint8'))
            prev_frame = gray_frame

        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


def faster_lucas_kanade_step(I1: np.ndarray,
                             I2: np.ndarray,
                             window_size: int) -> tuple[np.ndarray, np.ndarray]:
    du = np.zeros(I1.shape)
    dv = np.zeros(I1.shape)
    """INSERT YOUR CODE HERE. Calculate du and dv correctly"""
    # FACTOR = 5
    # if min(I1.shape) < FACTOR * window_size:
    #     return my_lucas_kanade_step(I1, I2, window_size)
    # else:
    h, w = I1.shape
    haris_response = cv2.cornerHarris(src=np.float32(I2), blockSize=5, k=0.05, ksize=3)
    corners = np.where(haris_response > 0.01 * haris_response.max())
    corners = np.transpose(corners)
    # Step1:
    Ix = signal.convolve2d(in1=I2, in2=X_DERIVATIVE_FILTER, mode='same', boundary='symm')
    Iy = signal.convolve2d(in1=I2, in2=Y_DERIVATIVE_FILTER, mode='same', boundary='symm')
    # Step2:
    It = I2 - I1
    # Step3:
    Ixx = Ix * Ix
    Iyy = Iy * Iy
    Ixy = Ix * Iy
    Ixt = -1 * Ix * It
    Iyt = -1 * Iy * It

    # Compute sliding windows for the arrays
    window_shape = (window_size, window_size)
    Ixx_windows = np.lib.stride_tricks.sliding_window_view(Ixx, window_shape)
    Iyy_windows = np.lib.stride_tricks.sliding_window_view(Iyy, window_shape)
    Ixy_windows = np.lib.stride_tricks.sliding_window_view(Ixy, window_shape)
    Ixt_windows = np.lib.stride_tricks.sliding_window_view(Ixt, window_shape)
    Iyt_windows = np.lib.stride_tricks.sliding_window_view(Iyt, window_shape)

    # Compute windows only around interest points
    y, x = corners[:, 0], corners[:, 1]
    y, x = np.clip(y, 0, Ixx_windows.shape[0]-1), np.clip(x, 0, Ixx_windows.shape[1]-1)
    Ixx_windows_at_interest_points = Ixx_windows[y, x, :, :].reshape((len(corners), window_size, window_size, 1, 1))
    Iyy_windows_at_interest_points = Iyy_windows[y, x, :, :].reshape((len(corners), window_size, window_size, 1, 1))
    Ixy_windows_at_interest_points = Ixy_windows[y, x, :, :].reshape((len(corners), window_size, window_size, 1, 1))
    Ixt_windows_at_interest_points = Ixt_windows[y, x, :, :].reshape((len(corners), window_size, window_size, 1, 1))
    Iyt_windows_at_interest_points = Iyt_windows[y, x, :, :].reshape((len(corners), window_size, window_size, 1, 1))

    # Calculate ATA and b using vectorized operations
    ATA_11 = np.sum(Ixx_windows_at_interest_points, axis=(1, 2, 3))
    ATA_12 = np.sum(Ixy_windows_at_interest_points, axis=(1, 2, 3))
    ATA_21 = ATA_12
    ATA_22 = np.sum(Iyy_windows_at_interest_points, axis=(1, 2, 3))
    ATA = np.stack([np.stack([ATA_11, ATA_12], axis=-1), np.stack([ATA_21, ATA_22], axis=-1)], axis=-2)
    b_1 = np.sum(Ixt_windows_at_interest_points, axis=(1, 2, 3))[:, :, np.newaxis]
    b_2 = np.sum(Iyt_windows_at_interest_points, axis=(1, 2, 3))[:, :, np.newaxis]
    b = np.concatenate([b_1, b_2], axis=-1)

    # Calculate U_V_LS using vectorized operations
    U_V_LS = np.zeros((h, w, 2))
    U_V_LS[y, x] = np.linalg.solve(ATA, b).squeeze()
    du, dv = U_V_LS[:, :, 0], U_V_LS[:, :, 1]
    return du, dv

def faster_lucas_kanade_optical_flow(
        I1: np.ndarray, I2: np.ndarray, window_size: int, max_iter: int,
        num_levels: int) -> tuple[np.ndarray, np.ndarray]:
    """Calculate LK Optical Flow for max iterations in num-levels .
    Use faster_lucas_kanade_step instead of lucas_kanade_step.
    Args:
        I1: np.ndarray. Image at time t.
        I2: np.ndarray. Image at time t+1.
        window_size: int. The window is of shape window_size X window_size.
        max_iter: int. Maximal number of LK-steps for each level of the pyramid.
        num_levels: int. Number of pyramid levels.
    Returns:
        (u, v): tuple of np.ndarray-s. Each one of the shape of the
        original image. v encodes the shift in rows and u in columns.
    """
    DOWN_FACTOR = 2
    h_factor = int(np.ceil(I1.shape[0] / (2 ** num_levels)))
    w_factor = int(np.ceil(I1.shape[1] / (2 ** num_levels)))
    IMAGE_SIZE = (w_factor * (2 ** num_levels),
                  h_factor * (2 ** num_levels))
    if I1.shape != IMAGE_SIZE:
        I1 = cv2.resize(I1, IMAGE_SIZE)
    if I2.shape != IMAGE_SIZE:
        I2 = cv2.resize(I2, IMAGE_SIZE)
    pyramid_I1 = build_pyramid(I1, num_levels)  # create levels list for I1
    pyarmid_I2 = build_pyramid(I2, num_levels)  # create levels list for I1
    u = np.zeros(pyarmid_I2[-1].shape)  # create u in the size of smallest image
    v = np.zeros(pyarmid_I2[-1].shape)  # create v in the size of smallest image
    """INSERT YOUR CODE HERE.
    Replace u and v with their true value."""
    for level in range(num_levels, -1, -1):
        I2_warp = my_warp_image(pyarmid_I2[level], u, v)
        for iter in range(max_iter):
            du, dv = faster_lucas_kanade_step(I1=pyramid_I1[level], I2=I2_warp, window_size=window_size)
            u += du
            v += dv
            I2_warp = my_warp_image(pyarmid_I2[level], u, v)
        if level > 0:
            h_scale, w_scale = pyarmid_I2[level - 1].shape
            u = cv2.resize(u, (w_scale, h_scale)) * DOWN_FACTOR
            v = cv2.resize(v, (w_scale, h_scale)) * DOWN_FACTOR
    return u, v

def lucas_kanade_faster_video_stabilization_fix_effects(
        input_video_path: str, output_video_path: str, window_size: int,
        max_iter: int, num_levels: int, start_rows: int = 10,
        start_cols: int = 2, end_rows: int = 30, end_cols: int = 30) -> None:
    """Calculate LK Optical Flow to stabilize the video and save it to file.
    Args:
        input_video_path: str. path to input video.
        output_video_path: str. path to output stabilized video.
        window_size: int. The window is of shape window_size X window_size.
        max_iter: int. Maximal number of LK-steps for each level of the pyramid.
        num_levels: int. Number of pyramid levels.
        start_rows: int. The number of lines to cut from top.
        end_rows: int. The number of lines to cut from bottom.
        start_cols: int. The number of columns to cut from left.
        end_cols: int. The number of columns to cut from right.
    Returns:
        None.
    """
    """INSERT YOUR CODE HERE."""
    cap = cv2.VideoCapture(input_video_path)
    params = get_video_parameters(cap)
    out = cv2.VideoWriter(output_video_path, fourcc=cv2.VideoWriter_fourcc(*'XVID'), fps=params["fps"],
                          frameSize=(params["width"], params["height"]), isColor=False)
    ret, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    out.write(gray_frame[start_rows:gray_frame.shape[0]-end_rows, start_cols:gray_frame.shape[1]-end_cols])

    h_factor = int(np.ceil(gray_frame.shape[0] / (2 ** (num_levels - 1 + 1))))
    w_factor = int(np.ceil(gray_frame.shape[1] / (2 ** (num_levels - 1 + 1))))
    IMAGE_SIZE = (w_factor * (2 ** (num_levels - 1 + 1)), h_factor * (2 ** (num_levels - 1 + 1)))
    gray_frame = cv2.resize(gray_frame, IMAGE_SIZE)
    u = np.zeros(gray_frame.shape, dtype=np.float)
    v = np.zeros(gray_frame.shape, dtype=np.float)
    prev_frame = gray_frame
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_frame = cv2.resize(gray_frame, IMAGE_SIZE)
            du, dv = faster_lucas_kanade_optical_flow(I1=prev_frame, I2=gray_frame, window_size=window_size, max_iter=max_iter, num_levels=num_levels)
            r_low_u, r_high_u = window_size // 2, du.shape[0] - window_size // 2
            c_low_u, c_high_u = window_size // 2, du.shape[1] - window_size // 2
            r_low_v, r_high_v = window_size // 2, dv.shape[0] - window_size // 2
            c_low_v, c_high_v = window_size // 2, dv.shape[1] - window_size // 2
            du_mean, dv_mean = np.mean(du[r_low_u:r_high_u, c_low_u:c_high_u]), np.mean(
                dv[r_low_v:r_high_v, c_low_v:c_high_v])
            # Part D
            u[r_low_u:r_high_u, c_low_u:c_high_u] += du_mean
            v[r_low_v:r_high_v, c_low_v:c_high_v] += dv_mean
            # Part E
            warp_frame = my_warp_image(gray_frame, u, v)
            warp_frame = warp_frame[start_rows:gray_frame.shape[0]-end_rows, start_cols:gray_frame.shape[1]-end_cols]
            warp_frame = cv2.resize(warp_frame, (params["width"], params["height"]))
            out.write(warp_frame.astype('uint8'))
            prev_frame = gray_frame

        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
