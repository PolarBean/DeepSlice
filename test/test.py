import DeepSlice
FILE = r'/home/harry/Documents/3/'
DS = DeepSlice.Model('mouse')
DS.predict(FILE, ensemble=False)
DS.enforce_index_spacing(thickness = 1)
print('predictions complete')
print('-----------------------------------------')
print('deforming')
DS.deform_atlas()
print('deforming complete')
print('-----------------------------------------')
DS.save_predictions(FILE + 'output')



