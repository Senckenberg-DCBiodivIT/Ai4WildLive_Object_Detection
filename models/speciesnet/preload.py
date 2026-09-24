from speciesnet import DEFAULT_MODEL, SpeciesNet

print(f"[preload] downloading speciesnet weights: {DEFAULT_MODEL}")
SpeciesNet(DEFAULT_MODEL)
print("[preload] speciesnet weights ready")
