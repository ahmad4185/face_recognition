
import FeatureVector as fv
import csvLibrary as csl


def generateFeatureVectorCSV(CATEGORIES, DATADIR, IMG_SIZE, ENABLE_RANDOM, CSV_FILE_NAME):
	dataset = fv.DataMatrixPreparation(CATEGORIES, DATADIR, IMG_SIZE, ENABLE_RANDOM)
	csl.generateCSV(dataset, CSV_FILE_NAME)