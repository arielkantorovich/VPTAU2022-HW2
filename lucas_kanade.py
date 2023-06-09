import cv2
import numpy as np
from tqdm import tqdm
from scipy import signal
from scipy.interpolate import griddata

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
    return np.array(pyramid, dtype=type(img_lev))

def lucas_kanade_step(I1: np.ndarray,
                      I2: np.ndarray,
                      window_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Perform one Lucas-Kanade Step.

    This method receives two images as inputs and a window_size. It
    calculates the per-pixel shift in the x-axis and y-axis. That is,
    it outputs two maps of the shape of the input images. The first map
    encodes the per-pixel optical flow parameters in the x-axis and the
    second in the y-axis.

    (1) Calculate Ix and Iy by convolving I2 with the appropriate filters (
    see the constants in the head of this file).
    (2) Calculate It from I1 and I2.
    (3) Calculate du and dv for each pixel:
      (3.1) Start from all-zeros du and dv (each one) of size I1.shape.
      (3.2) Loop over all pixels in the image (you can ignore boundary pixels up
      to ~window_size/2 pixels in each side of the image [top, bottom,
      left and right]).
      (3.3) For every pixel, pretend the pixel’s neighbors have the same (u,
      v). This means that for NxN window, we have N^2 equations per pixel.
      (3.4) Solve for (u, v) using Least-Squares solution. When the solution
      does not converge, keep this pixel's (u, v) as zero.
    For detailed Equations reference look at slides 4 & 5 in:
    http://www.cse.psu.edu/~rtc12/CSE486/lecture30.pdf

    Args:
        I1: np.ndarray. Image at time t.
        I2: np.ndarray. Image at time t+1.
        window_size: int. The window is of shape window_size X window_size.

    Returns:
        (du, dv): tuple of np.ndarray-s. Each one is of the shape of the
        original image. dv encodes the optical flow parameters in rows and du
        in columns.
    """
    """INSERT YOUR CODE HERE"""
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
    for i in range(window_size // 2, h - window_size // 2):
        for j in range(window_size // 2, w - window_size // 2):
            r_lower, r_upper = i - window_size // 2, i + 1 + window_size // 2
            c_lower, c_upper = j - window_size // 2, j + 1 + window_size // 2
            A = np.stack((Ix[r_lower:r_upper, c_lower:c_upper].reshape(-1),
                          Iy[r_lower:r_upper, c_lower:c_upper].reshape(-1)),
                         axis=-1)
            # Check solution converge
            C = A.T @ A
            if np.linalg.det(C) > epsilon:
                b = -It[r_lower:r_upper, c_lower:c_upper].reshape(-1, 1)
                U_V_LS = np.linalg.inv(C) @ A.T @ b
                du[i, j] = U_V_LS[0, 0]
                dv[i, j] = U_V_LS[1, 0]
    return du, dv


def warp_image(image: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Warp image using the optical flow parameters in u and v.

    Note that this method needs to support the case where u and v shapes do
    not share the same shape as of the image. We will update u and v to the
    shape of the image. The way to do it, is to:
    (1) cv2.resize to resize the u and v to the shape of the image.
    (2) Then, normalize the shift values according to a factor. This factor
    is the ratio between the image dimension and the shift matrix (u or v)
    dimension (the factor for u should take into account the number of columns
    in u and the factor for v should take into account the number of rows in v).

    As for the warping, use `scipy.interpolate`'s `griddata` method. Define the
    grid-points using a flattened version of the `meshgrid` of 0:w-1 and 0:h-1.
    The values here are simply image.flattened().
    The points you wish to interpolate are, again, a flattened version of the
    `meshgrid` matrices - don't forget to add them v and u.
    Use `np.nan` as `griddata`'s fill_value.
    Finally, fill the nan holes with the source image values.
    Hint: For the final step, use np.isnan(image_warp).

    Args:
        image: np.ndarray. Image to warp.
        u: np.ndarray. Optical flow parameters corresponding to the columns.
        v: np.ndarray. Optical flow parameters corresponding to the rows.

    Returns:
        image_warp: np.ndarray. Warped image.
    """
    image_warp = image.copy()
    h, w = image.shape
    """INSERT YOUR CODE HERE"""
    # Step1:
    U_FACTOR = w / u.shape[1]
    V_FACTOR = h / v.shape[0]
    u = cv2.resize(u, (w, h), interpolation=cv2.INTER_LINEAR) * U_FACTOR
    v = cv2.resize(v, (w, h), interpolation=cv2.INTER_LINEAR) * V_FACTOR
    # Step 2:
    # (2.1)
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    points = np.column_stack((x.flatten(), y.flatten()))
    x_new = x.flatten() + u.flatten()
    y_new = y.flatten() + v.flatten()
    points_new = np.column_stack((x_new, y_new))
    # (2.3) + (2.2)
    image_warp = griddata(points=points, values=image.flatten(),
                      xi=points_new, method='cubic', fill_value=np.nan)
    # (2.4) Handle with holes
    image_warp[np.isnan(image_warp)] = image.flatten()[np.isnan(image_warp)]
    # Reshape to the original shape
    image_warp = image_warp.reshape((h, w))
    return image_warp


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
        I2_level = pyarmid_I2[level]
        I1_level = pyramid_I1[level]
        I2_warp = warp_image(I2_level, u, v)
        for iter in range(max_iter):
            du, dv = lucas_kanade_step(I1=I1_level, I2=I2_warp, window_size=window_size)
            u += du
            v += dv
            I2_warp = warp_image(I2_level, u, v)
        if level > 0:
            h_scale, w_scale = pyarmid_I2[level - 1].shape
            u = cv2.resize(u, (w_scale, h_scale)) * DOWN_FACTOR
            v = cv2.resize(v, (w_scale, h_scale)) * DOWN_FACTOR
    return u, v


def lucas_kanade_video_stabilization(input_video_path: str,
                                     output_video_path: str,
                                     window_size: int,
                                     max_iter: int,
                                     num_levels: int) -> None:
    """Use LK Optical Flow to stabilize the video and save it to file.

    Args:
        input_video_path: str. path to input video.
        output_video_path: str. path to output stabilized video.
        window_size: int. The window is of shape window_size X window_size.
        max_iter: int. Maximal number of LK-steps for each level of the pyramid.
        num_levels: int. Number of pyramid levels.

    Returns:
        None.

    Recipe:
        (1) Open a VideoCapture object of the input video and read its
        parameters.
        (2) Create an output video VideoCapture object with the same
        parameters as in (1) in the path given here as input.
        (3) Convert the first frame to grayscale and write it as-is to the
        output video.
        (4) Resize the first frame as in the Full-Lucas-Kanade function to
        K * (2^(num_levels - 1)) X M * (2^(num_levels - 1)).
        Where: K is the ceil(h / (2^(num_levels - 1)),
        and M is ceil(h / (2^(num_levels - 1)).
        (5) Create a u and a v which are og the size of the image.
        (6) Loop over the frames in the input video (use tqdm to monitor your
        progress) and:
          (6.1) Resize them to the shape in (4).
          (6.2) Feed them to the lucas_kanade_optical_flow with the previous
          frame.
          (6.3) Use the u and v maps obtained from (6.2) and compute their
          mean values over the region that the computation is valid (exclude
          half window borders from every side of the image).
          (6.4) Update u and v to their mean values inside the valid
          computation region.
          (6.5) Add the u and v shift from the previous frame diff such that
          frame in the t is normalized all the way back to the first frame.
          (6.6) Save the updated u and v for the next frame (so you can
          perform step 6.5 for the next frame.
          (6.7) Finally, warp the current frame with the u and v you have at
          hand.
          (6.8) We highly recommend you to save each frame to a directory for
          your own debug purposes. Erase that code when submitting the exercise.
       (7) Do not forget to gracefully close all VideoCapture and to destroy
       all windows.
    """
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
            warp_frame = warp_image(gray_frame, u, v)
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
    """Faster implementation of a single Lucas-Kanade Step.

    (1) If the image is small enough (you need to design what is good
    enough), simply return the result of the good old lucas_kanade_step
    function.
    (2) Otherwise, find corners in I2 and calculate u and v only for these
    pixels.
    (3) Return maps of u and v which are all zeros except for the corner
    pixels you found in (2).

    Args:
        I1: np.ndarray. Image at time t.
        I2: np.ndarray. Image at time t+1.
        window_size: int. The window is of shape window_size X window_size.

    Returns:
        (du, dv): tuple of np.ndarray-s. Each one of the shape of the
        original image. dv encodes the shift in rows and du in columns.
    """

    du = np.zeros(I1.shape)
    dv = np.zeros(I1.shape)
    """INSERT YOUR CODE HERE. Calculate du and dv correctly"""
    FACTOR = 4
    if min(I1.shape) < FACTOR * window_size:
        return lucas_kanade_step(I1, I2, window_size)
    else:
        haris_response = cv2.cornerHarris(src=np.float32(I2), blockSize=5, k=0.05, ksize=3)
        corners = np.where(haris_response > 0.01 * haris_response.max())
        for i, j in zip(corners[0], corners[1]):
            r_lower, r_upper = max(0, i - window_size // 2), min(I1.shape[0], i + 1 + window_size // 2)
            c_lower, c_upper = max(0, j - window_size // 2), min(I1.shape[1], j + 1 + window_size // 2)
            I1_win = I1[r_lower:r_upper, c_lower:c_upper]
            I2_win = I2[r_lower:r_upper, c_lower:c_upper]
            # Step1
            Ix = signal.convolve2d(in1=I2_win, in2=X_DERIVATIVE_FILTER, mode='same', boundary='symm')
            Iy = signal.convolve2d(in1=I2_win, in2=Y_DERIVATIVE_FILTER, mode='same', boundary='symm')
            # Step2:
            It = I2_win - I1_win
            # Step3:
            A = np.stack((Ix.reshape(-1),
                                  Iy.reshape(-1)),
                                 axis=-1)
            b = -It.reshape(-1, 1)
            U_V_LS = np.linalg.inv(A.T @ A) @ A.T @ b
            du[i, j] = U_V_LS[0, 0]
            dv[i, j] = U_V_LS[1, 0]
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
        I2_warp = warp_image(pyarmid_I2[level], u, v)
        for iter in range(max_iter):
            du, dv = faster_lucas_kanade_step(I1=pyramid_I1[level], I2=I2_warp, window_size=window_size)
            u += du
            v += dv
            I2_warp = warp_image(pyarmid_I2[level], u, v)
        if level > 0:
            h_scale, w_scale = pyarmid_I2[level - 1].shape
            U_FACTOR = w_scale / u.shape[1]
            V_FACTOR = h_scale / v.shape[0]
            u = cv2.resize(u, (w_scale, h_scale)) * U_FACTOR
            v = cv2.resize(v, (w_scale, h_scale)) * V_FACTOR
    return u, v


def lucas_kanade_faster_video_stabilization(
        input_video_path: str, output_video_path: str, window_size: int,
        max_iter: int, num_levels: int) -> None:
    """Calculate LK Optical Flow to stabilize the video and save it to file.

    Args:
        input_video_path: str. path to input video.
        output_video_path: str. path to output stabilized video.
        window_size: int. The window is of shape window_size X window_size.
        max_iter: int. Maximal number of LK-steps for each level of the pyramid.
        num_levels: int. Number of pyramid levels.

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
            du, dv = faster_lucas_kanade_optical_flow(I1=prev_frame, I2=gray_frame, window_size=window_size,
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
            warp_frame = warp_image(gray_frame, u, v)
            warp_frame = cv2.resize(warp_frame, (params["width"], params["height"]))
            out.write(warp_frame.astype('uint8'))
            prev_frame = gray_frame

        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


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
            warp_frame = warp_image(gray_frame, u, v)
            warp_frame = warp_frame[start_rows:gray_frame.shape[0]-end_rows, start_cols:gray_frame.shape[1]-end_cols]
            warp_frame = cv2.resize(warp_frame, (params["width"], params["height"]))
            out.write(warp_frame.astype('uint8'))
            prev_frame = gray_frame

        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


