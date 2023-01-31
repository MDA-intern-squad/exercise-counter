import util

csv_loader = util.CSVLoader()
embeder = util.PoseEmbedderByDistance()

up = csv_loader('./data/test/up.csv')
down = csv_loader('./data/test/down.csv')

print(embeder(up))

