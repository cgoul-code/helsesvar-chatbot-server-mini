from PIL import Image

# Open JPG
img = Image.open("Forside-Banner_digiung.jpg")

# Save as EPS
img.save("Forside-Banner_digiung.eps", format="EPS")