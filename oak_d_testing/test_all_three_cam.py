import cv2
import depthai as dai
import numpy as np

RGB_SIZE = (1280, 720)
MONO_SIZE = (1280, 720)

with dai.Pipeline() as pipeline:
    # RGB camera (usually CAM_A)
    cam_rgb = pipeline.create(dai.node.Camera).build(boardSocket=dai.CameraBoardSocket.CAM_A)
    q_rgb = cam_rgb.requestOutput(RGB_SIZE, type=dai.ImgFrame.Type.BGR888p).createOutputQueue()

    # Left mono (usually CAM_B)
    cam_left = pipeline.create(dai.node.Camera).build(boardSocket=dai.CameraBoardSocket.CAM_B)
    q_left = cam_left.requestOutput(MONO_SIZE, type=dai.ImgFrame.Type.GRAY8).createOutputQueue()

    # Right mono (usually CAM_C)
    cam_right = pipeline.create(dai.node.Camera).build(boardSocket=dai.CameraBoardSocket.CAM_C)
    q_right = cam_right.requestOutput(MONO_SIZE, type=dai.ImgFrame.Type.GRAY8).createOutputQueue()

    pipeline.start()

    last_rgb = None
    last_left = None
    last_right = None

    while pipeline.isRunning():
        # Non-blocking reads so one slow stream doesn't stall the others
        msg = q_rgb.tryGet()
        if msg is not None:
            last_rgb = msg.getCvFrame()

        msg = q_left.tryGet()
        if msg is not None:
            last_left = msg.getCvFrame()

        msg = q_right.tryGet()
        if msg is not None:
            last_right = msg.getCvFrame()

        # Show only when we have at least one frame from each stream
        if last_rgb is not None and last_left is not None and last_right is not None:
            # Convert mono (1-channel) to BGR (3-channel) so hstack works
            left_bgr = cv2.cvtColor(last_left, cv2.COLOR_GRAY2BGR)
            right_bgr = cv2.cvtColor(last_right, cv2.COLOR_GRAY2BGR)

            # Make sure all frames are the same size (use RGB size as reference)
            h, w = last_rgb.shape[:2]
            if left_bgr.shape[:2] != (h, w):
                left_bgr = cv2.resize(left_bgr, (w, h), interpolation=cv2.INTER_AREA)
            if right_bgr.shape[:2] != (h, w):
                right_bgr = cv2.resize(right_bgr, (w, h), interpolation=cv2.INTER_AREA)

            side_by_side = np.hstack([left_bgr, last_rgb, right_bgr])
            cv2.imshow("OAK Cameras (Left | RGB | Right)", side_by_side)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cv2.destroyAllWindows()

