import csv
import re

def extract_info(file_path, output_path):
    with open(file_path, 'r') as file, open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(['Label', 'SortedList_First_Element', 'SortedList_Second_Element'])
        
        first_element = None
        second_element = None
        
        for line in file:
            line = line.strip()
            
            # Check if the line contains "currentIndex="
            if line.startswith("benignLogits= "):
                match = re.search(r'benignLogits=  \[([0-9e.-]+), ([0-9e.-]+), ', line)
                if match:
                    first_element = match.group(1)
                    second_element = match.group(2)
                    
            # Check if the line starts with "HSJA logits"
            elif line.startswith("lable =  tensor"):
                match = re.search(r'lable =  tensor\(\[(\d+)\], ', line)
                if match:
                    label_value = match.group(1)
                    writer.writerow([label_value, first_element, second_element])

if __name__ == "__main__":
    file_path = 'imagenet_benign.out'
    output_path = 'imagenet-benign-data.csv'
    extract_info(file_path, output_path)
