import detect

VIDEO_PATH = 'test2.mp4'

def object_detection():
    detect.main(detect.parse_opt())

if __name__ == '__main__':
    with open("logfaces.json","w") as f:
        f.write("[]")
    object_detection()