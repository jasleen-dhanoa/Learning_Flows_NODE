from PIL import Image
png_count = 5000
files = []
for i in range(50,png_count,50):
    seq = str(i)
    # file_names = 'Images/Inv_Single_Gyre/pred_' + seq + '.png'
    file_names = 'Images/Inv_Single_Gyre/vector_fields_' + seq + '.png'
    files.append(file_names)
print(files)
frames = []

for i in files:
    new_frame = Image.open(i)
    frames.append(new_frame)

# Save into a GIF file that loops forever
frames[0].save('SG_Inv_Vec_Fields.gif', format='GIF',
               append_images=frames[1:],
               save_all=True,
               duration=400, loop=0)