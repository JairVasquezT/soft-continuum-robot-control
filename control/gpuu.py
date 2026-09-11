import torch

# 1. Verificación principal
cuda_disponible = torch.cuda.is_available()
print(f"¿CUDA disponible?: {cuda_disponible}")

if cuda_disponible:
    # 2. Información del hardware detectado
    print(f"Número de GPUs disponibles: {torch.cuda.device_count()}")
    print(f"Nombre de la GPU: {torch.cuda.get_device_name(0)}")
    print(f"Versión de CUDA usada por PyTorch: {torch.version.cuda}")
    
    # 3. Prueba de fuego: Crear un tensor directamente en la GPU
    x = torch.rand(3, 3, device="cuda")
    print("\n✅ Tensor creado exitosamente en la GPU:")
    print(x)
else:
    print("\n⚠️ PyTorch NO está utilizando la GPU. Todo se ejecutará en la CPU.")