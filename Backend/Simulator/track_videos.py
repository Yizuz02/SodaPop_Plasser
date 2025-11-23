"""
Track Video Generator
Genera videos MP4 a partir de imágenes estáticas de vías,
creando un efecto de movimiento continuo hacia abajo.
"""

import cv2
import numpy as np
import subprocess
import tempfile
import os
from pathlib import Path


def generate_track_video_from_image(
    image_path: str,
    output_path: str,
    duration: float = 15.0,
    fps: int = 30,
    output_width: int = 720,
    output_height: int = 480,
    speed: float = 1.0,
):
    """
    Genera un video de movimiento continuo a partir de una imagen de vías.

    Args:
        image_path: Ruta a la imagen de la vía
        output_path: Ruta donde guardar el video MP4
        duration: Duración del video en segundos
        fps: Frames por segundo
        output_width: Ancho del video de salida
        output_height: Alto del video de salida
        speed: Multiplicador de velocidad (1.0 = normal)
    """
    print(f"Cargando imagen: {image_path}")

    # Cargar imagen
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"No se pudo cargar la imagen: {image_path}")

    img_height, img_width = img.shape[:2]
    print(f"Tamaño original: {img_width}x{img_height}")

    # Calcular el aspect ratio objetivo
    target_aspect = output_width / output_height
    img_aspect = img_width / img_height

    # Redimensionar imagen - ZOOM OUT (más pequeña = vista más amplia)
    # Usar un factor de escala menor para ver más de la imagen
    zoom_out_factor = 0.85  # Menor = más alejado

    if img_aspect > target_aspect:
        new_height = int(img_height * 2 * zoom_out_factor)
        new_width = int(new_height * img_aspect)
    else:
        new_width = int(output_width * zoom_out_factor)
        new_height = int(new_width / img_aspect)

    # Asegurar que la imagen sea lo suficientemente alta para el scroll
    min_height = output_height * 4
    if new_height < min_height:
        scale = min_height / new_height
        new_height = min_height
        new_width = int(new_width * scale)

    # Redimensionar
    img_resized = cv2.resize(
        img, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4
    )

    # Crear imagen extendida (tile vertical para loop seamless)
    # Concatenamos la imagen consigo misma para un loop suave
    img_tiled = np.vstack([img_resized, img_resized, img_resized])
    tiled_height = img_tiled.shape[0]

    print(f"Imagen procesada: {new_width}x{tiled_height}")

    # Calcular parámetros de animación
    total_frames = int(duration * fps)
    pixels_per_frame = (img_resized.shape[0] * speed) / (fps * 2)  # Velocidad de scroll

    print(f"Generando {total_frames} frames a {fps} FPS...")

    # Crear archivo temporal para el video raw
    temp_raw = tempfile.NamedTemporaryFile(suffix=".avi", delete=False)
    temp_raw_path = temp_raw.name
    temp_raw.close()

    # Inicializar VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out = cv2.VideoWriter(temp_raw_path, fourcc, fps, (output_width, output_height))

    # Centrar horizontalmente
    x_offset = max(0, (new_width - output_width) // 2)

    # Generar frames
    for frame_num in range(total_frames):
        # Calcular posición Y del scroll (con loop)
        y_offset = int((frame_num * pixels_per_frame) % img_resized.shape[0])

        # Extraer región visible
        frame = img_tiled[
            y_offset : y_offset + output_height, x_offset : x_offset + output_width
        ].copy()

        # Asegurar que el frame tenga el tamaño correcto
        if frame.shape[0] != output_height or frame.shape[1] != output_width:
            frame = cv2.resize(frame, (output_width, output_height))

        # Agregar efectos sutiles para que parezca más "video"

        # 1. Ligero ruido de sensor de cámara
        if frame_num % 2 == 0:
            noise = np.random.randint(-3, 4, frame.shape, dtype=np.int16)
            frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # 2. Variación de brillo muy sutil (simula exposición automática)
        brightness_var = int(3 * np.sin(frame_num * 0.05))
        frame = np.clip(frame.astype(np.int16) + brightness_var, 0, 255).astype(
            np.uint8
        )

        # 3. Ligero motion blur vertical (simula movimiento)
        if speed > 0.8:
            kernel_size = max(3, int(speed * 2)) | 1  # Asegurar impar
            kernel = np.zeros((kernel_size, 1), np.float32)
            kernel[:, 0] = 1.0 / kernel_size
            frame = cv2.filter2D(frame, -1, kernel)

        out.write(frame)

        # Progreso
        if frame_num % (fps * 2) == 0:
            print(
                f"  Progreso: {frame_num}/{total_frames} frames ({100*frame_num//total_frames}%)"
            )

    out.release()
    print(f"Video raw generado: {temp_raw_path}")

    # Convertir a H.264 con ffmpeg
    print("Convirtiendo a H.264...")
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                temp_raw_path,
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "20",  # Mejor calidad
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                output_path,
            ],
            check=True,
            capture_output=True,
        )
        print(f"✅ Video guardado: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error en ffmpeg: {e.stderr.decode()}")
        # Fallback: copiar el raw
        import shutil

        shutil.copy(temp_raw_path, output_path)

    # Limpiar
    os.unlink(temp_raw_path)

    # Mostrar info del video final
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"📹 Tamaño final: {file_size:.2f} MB")

    return output_path


def main():
    """Genera los 3 videos de vías a partir de las imágenes"""

    # Configuración
    assets_dir = Path("images/")
    output_dir = Path("videos/")
    output_dir.mkdir(exist_ok=True)

    # Definir los videos a generar
    videos_config = [
        {
            "input": assets_dir / "via1.png",
            "output": output_dir / "track_video_1.mp4",
            "duration": 25.0,
            "speed": 0.5,  # Más lento
            "name": "Zona Boscosa",
        },
        {
            "input": assets_dir / "via2.png",
            "output": output_dir / "track_video_2.mp4",
            "duration": 25.0,
            "speed": 0.45,  # Más lento
            "name": "Balasto Marrón",
        },
        {
            "input": assets_dir / "via3.webp",
            "output": output_dir / "track_video_3.mp4",
            "duration": 25.0,
            "speed": 0.5,  # Más lento
            "name": "Balasto Gris",
        },
    ]

    print("=" * 60)
    print("GENERADOR DE VIDEOS DE VÍAS FERROVIARIAS")
    print("=" * 60)

    for i, config in enumerate(videos_config, 1):
        print(f"\n[{i}/3] Generando video: {config['name']}")
        print("-" * 40)

        generate_track_video_from_image(
            image_path=str(config["input"]),
            output_path=str(config["output"]),
            duration=config["duration"],
            speed=config["speed"],
            output_width=720,
            output_height=540,  # Más alto = vista más amplia
            fps=30,
        )

    print("\n" + "=" * 60)
    print("✅ TODOS LOS VIDEOS GENERADOS")
    print("=" * 60)

    # Listar archivos generados
    print("\nArchivos generados:")
    for f in output_dir.glob("*.mp4"):
        size = f.stat().st_size / (1024 * 1024)
        print(f"  📹 {f.name} ({size:.2f} MB)")


if __name__ == "__main__":
    main()
