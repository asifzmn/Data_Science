import cv2
import os

if __name__ == '__main__':
    image_folder = '/home/az/Desktop/Video Maker Python/Memes'
    video_name = image_folder+'.avi'
    each_image_duration = 3
    fourcc  = cv2.VideoWriter_fourcc(*'XVID')

    # for img in os.listdir(image_folder):cv2.imwrite(os.path.join(image_folder, img)[:-4]+".png", cv2.imread(os.path.join(image_folder, img), 1))


    images = [img for img in os.listdir(image_folder) if img.endswith(".png")][:15]
    # images = [img for img in os.listdir(image_folder) ]

    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = 500, 500, 3

    video = cv2.VideoWriter(video_name,fourcc, 1, (width,height))

    for image in images:
        imageFile = cv2.resize(cv2.imread(os.path.join(image_folder, image)), (width, height))
        # cv2.imshow('image', imageFile)
        # cv2.waitKey(0)
        for _ in range(each_image_duration):video.write(imageFile)

    cv2.destroyAllWindows()
    video.release()

    cap = cv2.VideoCapture(video_name)

    if (cap.isOpened() == False): print("Error opening video  file")

    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            cv2.imshow('Frame', frame)

            # Press Q on keyboard to  exit
            if cv2.waitKey(1000) & 0xFF == ord('q'):
                break
        else:break

    cap.release()
    # Closes all the frames
    cv2.destroyAllWindows()