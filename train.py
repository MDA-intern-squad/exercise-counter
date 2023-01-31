import util
import numpy as np

csv_loader = util.CSVLoader()

up = csv_loader('./data/test/up.csv')
down = csv_loader('./data/test/down.csv')

embeder = util.PoseEmbedderByAngle()

embeded_up = np.array([embeder(i) for i in up], dtype=np.float32)
embeded_down = np.array([embeder(i) for i in down], dtype=np.float32)

model = util.ModelComplier()
model.compile({
    1: embeded_up,
    0: embeded_down
})
model.save('./model.h5')