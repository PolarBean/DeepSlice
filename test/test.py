import DeepSlice
FILE = r'/examples/example_brain/GLTa'
DS = DeepSlice.Model('mouse')
DS.predict(FILE, ensemble=False)
DS.enforce_index_spacing(thickness = 1)
print('predictions complete')
print('-----------------------------------------')
DS.save_predictions(FILE + 'output')



