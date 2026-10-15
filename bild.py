import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Sökvägar till dina bilder
img1 = mpimg.imread("/Users/josefakravi/Confusion_RF_centrliserad.png")
img2 = mpimg.imread("/Users/josefakravi/Confusion_IF_centraliserad.png")

# Skapa 1x2 layout
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Vänster bild
axes[0].imshow(img1)
axes[0].axis("off")

# Höger bild
axes[1].imshow(img2)
axes[1].axis("off")

# Justera layout
plt.tight_layout()

# Spara ihopslagen bild
plt.savefig("/Users/josefakravi/combined.png", dpi=300)

plt.show()