import torch

def organize_output(output):
    category_boundary = [0, 7, 10, 13, 17, 23, 26]
    # output will be separated into six part, and find each part the biggest index as new output
    new_output = []
    for i in range(1, len(category_boundary)):
        lower_bound = category_boundary[i - 1]
        upper_bound = category_boundary[i]
        temp_output = output[:, lower_bound:upper_bound]
        temp_index = torch.argmax(temp_output, dim=1)
        new_output.append(temp_index)
    new_output = torch.cat(new_output, 0).reshape(6, -1).float()
    return new_output.T