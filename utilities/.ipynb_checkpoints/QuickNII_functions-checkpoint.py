import pandas as pd
import xml.etree.ElementTree as ET
from tqdm import tqdm
import re
# converts a pandas DataFrame to a quickNII compatible XML
def pd_to_quickNII(results, orientation='coronal', filename='Download', web=False, folder_name=None):
    
    # Replace the subfolder present in results if we are running for the web
    if web and folder_name:
        results["Filenames"] = results["Filenames"].str.replace(folder_name, '').str.replace('\\', '').str.replace('/', '')
    # Get the total number of sections
    num_of_sections = results.shape[0]
    # Create the XML structure
    root = ET.Element('series')
    root.attrib['first'] = "1"
    root.attrib['last'] = str(num_of_sections)
#     if orientation == 'coronal':
#         results = results.sort_values('oy')
    if orientation == 'sagittal':
        results = results.sort_values('ox')
    if orientation == 'horizontal':
        results = results.sort_values('oz')
    # Explicitly confirm all filenames are Strings
    results['Filenames'] = results['Filenames'].astype(str)
    # for each section append Oxyz, Uxyz and Vxyz parameters to the XML
    for i in tqdm(range(num_of_sections)):
        child = ET.SubElement(root, 'slice')
        # this is the filename in our results file
        child.attrib['filename'] = results.iloc[i, 0]
        root.attrib['name'] = results.iloc[i, 0]  # so is this
        # Organise our coordinates
        ox, oy, oz, ux, uy, uz, vx, vy, vz = results.iloc[i, 1:10]
        # these next two values I believe are placeholders required by QuickNII.
        child.attrib["height"] = "700"
        child.attrib["width"] = "700"
        # Section number
        child.attrib["nr"] = str(i)
        # writes Oxyz, Uxyz and Vxyz parameters to the XML in the correct format
        child.attrib['anchoring'] = 'ox=' + str(ox) + '&oy=' + str(oy) + '&oz=' + str(oz) + '&ux=' + str(
            ux) + '&uy=' + str(uy) + '&uz=' + str(uz) + '&vx=' + str(vx) + '&vy=' + str(vy) + '&vz=' + str(vz)
    ET.ElementTree(root).write('{}.xml'.format(filename))
    results.to_csv('{}.csv'.format(filename))

# a useful script that converts a Quicknii XML to a csv file
# handy for training on human-aligned QuickNII files.


def XML_to_csv(xml):
    tree = ET.parse(str(xml))
    root = tree.getroot()
    count = 0
    df = pd.DataFrame()
    for i in root.findall('slice'):
        try:
            stringdata = str(i.attrib['anchoring'])
        except KeyError:
            continue
        df.loc[count, 'Filenames'] = i.attrib['filename']
        df.loc[count, 'ox'] = re.search('ox=(.+?)&oy', stringdata).group(1)
        df.loc[count, 'oy'] = re.search('oy=(.+?)&oz', stringdata).group(1)
        df.loc[count, 'oz'] = re.search('oz=(.+?)&ux', stringdata).group(1)
        df.loc[count, 'ux'] = re.search('ux=(.+?)&uy', stringdata).group(1)
        df.loc[count, 'uy'] = re.search('uy=(.+?)&uz', stringdata).group(1)
        df.loc[count, 'uz'] = re.search('uz=(.+?)&vx', stringdata).group(1)
        df.loc[count, 'vx'] = re.search('vx=(.+?)&vy', stringdata).group(1)
        df.loc[count, 'vy'] = re.search('vy=(.+?)&vz', stringdata).group(1)
        df.loc[count, 'vz'] = re.search('vz=(.+?)$', stringdata).group(1)
        count += 1
    return(df)
