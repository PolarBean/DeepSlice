import DeepSlice
from glob import glob

DS = DeepSlice.Model("mouse")
files = glob(
    r"/mnt/c/users/harryc/Applications/QSlicer/brain_files/e9dbe6aecb8047bfba0da21b2abb90ce/"
)
for file in files:
    print(file)
    file += "/images/"
    DS.predict(file, ensemble=False)
    DS.propagate_angles()
    DS.save_predictions(file + "//raw")
    DS.enforce_index_order()
    DS.save_predictions(file + "//order")
    DS.enforce_index_spacing()
    DS.save_predictions(file + "//even")

