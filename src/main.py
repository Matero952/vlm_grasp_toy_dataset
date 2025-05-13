from run_experiment import *
from agents.claude import *
from agents.gemini import *
def main(model_list: list):
    results = []
    for model in model_list:
        if "claude" in model:
            acc, _c, _r = run_experiment(ClaudeExperiment(model, prompt="Using the least amount of tokens, what is the best grasp for this object based on Feix's grasp taxonomy from the paper: The GRASP taxonomy of human grasp types. IEEE Transactions on Human-Machine Systems?"), model=model)
        else:
            acc, _c, _r = run_experiment(GeminiExperiment(model, prompt="Using the least amount of tokens, what is the best grasp for this object based on Feix's grasp taxonomy from the paper: The GRASP taxonomy of human grasp types. IEEE Transactions on Human-Machine Systems?"), model=model)
        results.append({model : [acc, _c, _r]})
    print(results)

if __name__ == "__main__":
    main(["claude-3-haiku-20240307", "gemini-2.0-flash"])
