import csv

def get_train_data():
    with open('zad1.csv', 'rb') as mycsvfile:
        data = list(csv.reader(mycsvfile))
        data = data[1:]
        for row in data:
            for index, number in enumerate(row):
                row[index] = float(number)

    return data