import DeepSlice
FILE = r'/home/harry/Documents/3/'
DS = DeepSlice.Model('mouse')
DS.predict(FILE, ensemble=False)
DS.propagate_angles()

DS.enforce_index_spacing(section_thickness = 1)
DS.save_predictions(FILE + 'output')

print('predictions complete')
print('-----------------------------------------')
print('deforming')
DS.deform_atlas()
print('deforming complete')
print('-----------------------------------------')
DS.save_predictions(FILE + 'output')



