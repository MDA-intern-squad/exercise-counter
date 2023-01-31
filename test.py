import util

csv_loader = util.CSVLoader()
embeder = util.PoseEmbedderByDistance()

up = csv_loader('./data/test/up.csv')
down = csv_loader('./data/test/down.csv')

embeded = embeder(up[0])
print(embeded)
print(embeded.flatten())
print(embeded.reshape(23, 3))
