import cv2
import depthai as dai

with dai.Pipeline() as pipeline:
    cam = pipeline.create(dai.node.Camera).build(boardSocket=dai.CameraBoardSocket.CAM_B)  # left mono
    q = cam.requestOutput((1280, 720), type=dai.ImgFrame.Type.GRAY8).createOutputQueue()

    pipeline.start()
    while pipeline.isRunning():
        frame = q.get().getCvFrame()
        cv2.imshow("left", frame)
        if cv2.waitKey(1) == ord("q"):
            break

