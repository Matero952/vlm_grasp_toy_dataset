from agents.gemini import *
import pandas as pd
import time
import ast
def run_experiment(experiment, model):
    save_dir = f"results/{model}/"
    os.makedirs(save_dir, exist_ok=True)
    new_df_path = os.path.join(save_dir, f"{model}_results.csv")
    if os.path.exists(new_df_path):
        new_df = pd.read_csv(new_df_path)
    else:
        new_df = pd.DataFrame(
            columns=["img_path", "target_grasp", "pred_grasp", "text"]
        )
    correct = 0
    seen = 0
    counter = 0
    for img_pth in generate_img_path_list("data"):
        if img_pth in new_df["img_path"].values:
            to_check_row = new_df[new_df["img_path"] == img_pth]
            # print(to_check_row.iloc[0]["target_grasp"])
            # print(type(to_check_row.iloc[0]["target_grasp"]))
            # print(to_check_row.iloc[0]["pred_grasp"])
            # print(type(to_check_row.iloc[0]["pred_grasp"]))
            # print(True) if to_check_row.iloc[0]["pred_grasp"] in str(to_check_row.iloc[0]["target_grasp"]) else print(False)
            # breakpoint()
            if len(ast.literal_eval(to_check_row.iloc[0]["target_grasp"])) > 4:
                target = []
                target_str = ''.join(ast.literal_eval(to_check_row.iloc[0]["target_grasp"]))
                target.append(target_str)
                print(target)
            else:
                target = ast.literal_eval(to_check_row.iloc[0]["target_grasp"])
            if to_check_row.iloc[0]["pred_grasp"] in target:
                correct += 1
            seen += 1
            # print("skipping!")
            counter += 1
            print(f"Status: {counter}/{len(generate_img_path_list("data"))}; Accuracy: {correct}/{seen}")
            continue
        text, grasp = experiment.process_sample(img_pth)
        if grasp is None:
            pass
        else:
            grasp = grasp.group(0)
        target = list(get_ground_truth(img_pth))
        if grasp is not None:
            if grasp in target:
                correct += 1
            else:
                pass
        else:
            pass
        seen += 1
        new_df = pd.concat([new_df, pd.DataFrame(
            [[img_pth, str(target), grasp, text]], columns= ["img_path", "target_grasp", "pred_grasp", "text"]
        )])
        new_df.to_csv(new_df_path, index=False)
        counter += 1
        print(f"Status: {counter}/{len(generate_img_path_list("data"))}; Accuracy: {correct}/{seen}")
        time.sleep(5.01)
    return correct / seen, correct, seen
        
def generate_img_path_list(parent_dir):
    pths = []
    for root, dirs, files in os.walk('data'):
        for file in files:
            pths.append(os.path.join(root, file))
    return pths

def get_ground_truth(pths_list):
    if "c_card" in pths_list:
        return "lateral"
    elif "cd" in pths_list:
        return "power disk", "precision disk"
    elif "coin" in pths_list:
        return "palmar pinch"
    elif "golf" in pths_list:
        return "tripod", "inferior pincer", "quadpod"
    elif "plate" in pths_list:
        return "extension type", "palmar"
    elif "tennis" in pths_list:
        return "power sphere", "precision sphere", "sphere 3 finger", "sphere 4 finger"
    return None
    




if __name__ == "__main__":
    run_experiment(GeminiExperiment(model="gemini-2.0-flash", prompt="What is the best grasp for this object based on Feix's grasp taxonomy from the paper: The GRASP taxonomy of human grasp types. IEEE Transactions on Human-Machine Systems?"), "gemini-2.0-flash")

    