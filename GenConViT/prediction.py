import os
import argparse
import json
from time import perf_counter
from datetime import datetime
from model.pred_func import *
from model.config import load_config

config = load_config()

def vids(
    ed_weight, vae_weight, root_dir="sample_prediction_data", dataset=None, num_frames=15, net=None, fp16=False
):
    result = set_result()
    r = 0
    f = 0
    count = 0
    
    model = load_genconvit(config, net, ed_weight, vae_weight, fp16)

    if os.path.isfile(root_dir):
        curr_vid = root_dir
        try:
            is_vid_folder = is_video_folder(curr_vid)
            if is_video(curr_vid) or is_vid_folder:
                result, accuracy, count, pred = predict(
                    curr_vid,
                    model,
                    fp16,
                    result,
                    num_frames,
                    net,
                    "uncategorized",
                    count,
                    vid_folder=is_vid_folder
                )
                f, r = (f + 1, r) if "FAKE" == real_or_fake(pred[0]) else (f, r + 1)
                print(f"Prediction: {pred[1]} {real_or_fake(pred[0])} \t\tFake: {f} Real: {r}")
            else:
                print(f"Invalid video file: {curr_vid}. Please provide a valid video file.")
        except Exception as e:
            print(f"An error occurred: {str(e)}")
        return result

    for filename in os.listdir(root_dir):
        curr_vid = os.path.join(root_dir, filename)
        try:
            is_vid_folder = is_video_folder(curr_vid)
            if is_video(curr_vid) or is_vid_folder:
                result, accuracy, count, pred = predict(
                    curr_vid,
                    model,
                    fp16,
                    result,
                    num_frames,
                    net,
                    "uncategorized",
                    count,
                    vid_folder=is_vid_folder
                )
                f, r = (f + 1, r) if "FAKE" == real_or_fake(pred[0]) else (f, r + 1)
                print(f"Prediction: {pred[1]} {real_or_fake(pred[0])} \t\tFake: {f} Real: {r}")
            else:
                print(f"Invalid video file: {curr_vid}. Please provide a valid video file.")
        except Exception as e:
            print(f"An error occurred: {str(e)}")

    return result

def predict(
    vid,
    model,
    fp16,
    result,
    num_frames,
    net,
    klass,
    count=0,
    accuracy=-1,
    correct_label="unknown",
    compression=None,
    vid_folder=None
):
    count += 1
    print(f"\n\n{str(count)} Loading... {vid}")

    start_time = perf_counter()

    # locate the extracted frames of the video if provided.
    if vid_folder:
        df = df_face_from_folder(vid, num_frames)
    else:
        df = df_face(vid, num_frames)  # extract face from the frames

    if fp16:
        df.half()
    
    y, y_val = (
        pred_vid(df, model)
        if len(df) >= 1
        else (torch.tensor(0).item(), torch.tensor(0.5).item())
    )
    result = store_result(
        result, os.path.basename(vid), y, y_val, klass, correct_label, compression
    )

    if accuracy > -1:
        if correct_label == real_or_fake(y):
            accuracy += 1
        print(
            f"\nPrediction: {y_val} {real_or_fake(y)} \t\t {accuracy}/{count} {accuracy/count}"
        )

    end_time = perf_counter()
    print("\n\n only one video--- %s seconds ---" % (end_time - start_time))
    
    return result, accuracy, count, [y, y_val]


def gen_parser():
    parser = argparse.ArgumentParser("GenConViT prediction")
    parser.add_argument("--p", type=str, help="video or image path")
    parser.add_argument(
        "--f", type=int, help="number of frames to process for prediction"
    )
    parser.add_argument(
        "--d", type=str, help="dataset type, dfdc, faceforensics, timit, celeb"
    )
    parser.add_argument(
        "--s", help="model size type: tiny, large.",
    )
    parser.add_argument(
        "--e", nargs='?', const='genconvit_ed_inference', default='genconvit_ed_inference', help="weight for ed.",
    )
    parser.add_argument(
        "--v", '--value', nargs='?', const='genconvit_vae_inference', default='genconvit_vae_inference', help="weight for vae.",
    )
    
    parser.add_argument("--fp16", type=str, help="half precision support")

    args = parser.parse_args()
    path = args.p
    num_frames = args.f if args.f else 15
    dataset = args.d if args.d else "other"
    fp16 = True if args.fp16 else False

    net = 'genconvit'
    ed_weight = 'genconvit_ed_inference'
    vae_weight = 'genconvit_vae_inference'

    if args.e and args.v:
        ed_weight = args.e
        vae_weight = args.v
    elif args.e:
        net = 'ed'
        ed_weight = args.e
    elif args.v:
        net = 'vae'
        vae_weight = args.v
    
        
    print(f'\nUsing {net}\n')  
    

    if args.s:
        if args.s in ['tiny', 'large']:
            config["model"]["backbone"] = f"convnext_{args.s}"
            config["model"]["embedder"] = f"swin_{args.s}_patch4_window7_224"
            config["model"]["type"] = args.s
    
    return path, dataset, num_frames, net, fp16, ed_weight, vae_weight


def main():
    start_time = perf_counter()
    path, dataset, num_frames, net, fp16, ed_weight, vae_weight = gen_parser()
    result = vids(ed_weight, vae_weight, path, dataset, num_frames, net, fp16)

    curr_time = datetime.now().strftime("%B_%d_%Y_%H_%M_%S")
    file_path = os.path.join("GenConViT", "result", f"prediction_{dataset}_{net}_{curr_time}.json")
    
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, "w") as f:
        json.dump(result, f)
    end_time = perf_counter()
    print("\n\n--- %s seconds ---" % (end_time - start_time))


if __name__ == "__main__":
    main()
