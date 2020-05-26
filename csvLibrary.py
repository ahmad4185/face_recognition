
import csv

def generateCSV(dataset, csv_filename):
	with open(csv_filename, 'w', newline="") as csvfile:
	    filewriter = csv.writer(csvfile)
	    filewriter.writerows(dataset)
	print("File has been saved with the directory given below")
	print('\n\n', csv_filename)



