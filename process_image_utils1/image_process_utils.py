import controlnet_hinter

hint_func={
    'pose': controlnet_hinter.hint_openpose,
    'scribble': controlnet_hinter.hint_scribble,
    'canny': controlnet_hinter.hint_canny,
    'hed': controlnet_hinter.hint_hed,
    'depth': controlnet_hinter.hint_depth,
    'seg':controlnet_hinter.hint_segmentation,
    'normal':controlnet_hinter.hint_normal,
    'mlsd':controlnet_hinter.hint_hough,
}


def find_mode(hint, image):
    hint_fn = hint_func[hint]
    print(hint_fn)
    mode= hint_fn(image)
    return mode
