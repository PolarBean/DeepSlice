from DeepSlice import DeepSlice
import os
os.chdir(r'../../../Downloads')
DS = DeepSlice()
DS.Build()
DS.predict(r'ISH_71717640_Calb1-20220207T142248Z-001')
DS.set_angles(DV=6, ML=7)
DS.Save_Results(r'ISH_71717640_Calb1-20220207T142248Z-001//DS-test-manual-angles')
print('success')
