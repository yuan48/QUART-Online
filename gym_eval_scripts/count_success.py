import os
import csv
from openpyxl import Workbook

def count_success(result_path):
    data = {}
    print(f"Counting success in result path: {result_path}")
    for task_folder in os.listdir(result_path):
        task_path = os.path.join(result_path, task_folder)

        # Count success rate for each task
        total_count = 0
        success_count = 0
        import pdb; pdb.set_trace()
        for episode_folder in os.listdir(task_path): #task
            if episode_folder == "videos":
                continue
            env_path = os.path.join(task_path, episode_folder)

            if os.path.isdir(env_path):
                csv_file_path = os.path.join(env_path, 'info.csv')
                if os.path.exists(csv_file_path):
                    with open(csv_file_path, 'r') as file:
                        reader = csv.reader(file)
                        lines = []
                        for line in reader:
                            lines.append(line)

                    total_count += 1

                    if lines[1][-1]=='True':
                        success_count += 1

        data.setdefault(task_folder, [0, 0])
        data[task_folder][0] += total_count
        data[task_folder][1] += success_count

    # import pdb; pdb.set_trace()
    total_total_count = sum(data[task][0] for task in data)
    total_success_count = sum(data[task][1] for task in data)
    overall_success_rate = total_success_count / total_total_count if total_total_count > 0 else 0

    wb = Workbook()
    ws = wb.active
    ws.append(["Task name", "Total Count", "Success Count", "Success Rate"])

    for task_name, (total_count, success_count ) in data.items():
        success_rate = success_count / total_count if total_count > 0 else 0
        ws.append([task_name, total_count, success_count, f"{success_rate:.2%}" ])

    ws.append(["Overall", total_total_count, total_success_count, f"{overall_success_rate:.2%}"])

    excel_path = f'{result_path}/total_result.xlsx'
    
    wb.save(excel_path)
    print(f"Statistic excel saved in {result_path}/total_result.xlsx") 

if __name__ == '__main__':
    result_path='./output/result_path'
    count_success(result_path)