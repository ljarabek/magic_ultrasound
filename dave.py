def optical_flow(one, two):
    """
    method taken from (https://chatbotslife.com/autonomous-vehicle-speed-estimation-from-dashboard-cam-ca96c24120e4)
    """
    one_g = cv2.cvtColor(one, cv2.COLOR_RGB2GRAY)
    two_g = cv2.cvtColor(two, cv2.COLOR_RGB2GRAY)
    hsv = np.zeros((120, 320, 3))
    # set saturation
    hsv[:,:,1] = cv2.cvtColor(two, cv2.COLOR_RGB2HSV)[:,:,1]
    # obtain dense optical flow paramters
    flow = cv2.calcOpticalFlowFarneback(one_g, two_g, flow=None,
                                        pyr_scale=0.5, levels=1, winsize=15,
                                        iterations=2,
                                        poly_n=5, poly_sigma=1.1, flags=0)
    # convert from cartesian to polar
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    # hue corresponds to direction
    hsv[:,:,0] = ang * (180/ np.pi / 2)
    # value corresponds to magnitude
    hsv[:,:,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    # convert HSV to int32's
    hsv = np.asarray(hsv, dtype= np.float32)
    rgb_flow = cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)
    return rgb_flow


def optical_flow(I1g, I2g, window_size, tau=1e-2):
  kernel_x = np.array([[-1., 1.], [-1., 1.]])
  kernel_y = np.array([[-1., -1.], [1., 1.]])
  kernel_t = np.array([[1., 1.], [1., 1.]])  # *.25
  w = window_size / 2  # window_size is odd, all the pixels with offset in between [-w, w] are inside the window
  I1g = I1g / 255.  # normalize pixels
  I2g = I2g / 255.  # normalize pixels
  # Implement Lucas Kanade
  # for each point, calculate I_x, I_y, I_t
  mode = 'same'
  fx = signal.convolve2d(I1g, kernel_x, boundary='symm', mode=mode)
  fy = signal.convolve2d(I1g, kernel_y, boundary='symm', mode=mode)
  ft = signal.convolve2d(I2g, kernel_t, boundary='symm', mode=mode) + signal.convolve2d(I1g, -kernel_t, boundary='symm', mode=mode)
  signal.convolve2d(I1g, -kernel_t, boundary='symm', mode=mode)


  u = np.zeros(I1g.shape)
  v = np.zeros(I1g.shape)
  # within window window_size * window_size
  for i in range(w, I1g.shape[0] - w):
    for j in range(w, I1g.shape[1] - w):
      Ix = fx[i - w:i + w + 1, j - w:j + w + 1].flatten()
      Iy = fy[i - w:i + w + 1, j - w:j + w + 1].flatten()
      It = ft[i - w:i + w + 1, j - w:j + w + 1].flatten()
      # b = ... # get b here
      # A = ... # get A here
      # if threshold τ is larger than the smallest eigenvalue of A'A:
      nu = ...  # get velocity here
      u[i, j] = nu[0]
      v[i, j] = nu[1]

  return (u, v)