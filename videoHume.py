import cv2
import asyncio
import os
import json
import time
from hume import AsyncHumeClient
from hume.expression_measurement.stream import Config
from hume.expression_measurement.stream.socket_client import StreamConnectOptions

# Configuración
INTERVALO_SEGUNDOS = 1
TIEMPO_LIMITE_SEGUNDOS = 300  # 5 minutos
DURACION_VIDEO_MS = 4999
FPS = 10  # FPS fijo para control del número de frames
RESOLUCION = (640, 480)

# Emociones específicas a capturar
EMOCIONES_OBJETIVO = [
    "Amusement", "Anger", "Awe", "Concentration", "Confusion", "Contempt",
    "Contentment", "Desire", "Disappointment", "Doubt", "Elation", "Interest",
    "Pain", "Sadness", "Surprise Positive", "Surprise Negative", "Triumph",
    "Sympathy", "Shame", "Pride", "Fear"
]

# Variables globales
tiempos_procesamiento = []
errores_lectura = 0
errores_emociones = 0
fotogramas_enviados = 0
informe_detallado = []

async def process_video(client: AsyncHumeClient):
    global errores_lectura, errores_emociones, tiempos_procesamiento, fotogramas_enviados

    tiempo_inicio_total = time.time()
    model_config = Config(face={})
    stream_options = StreamConnectOptions(config=model_config)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Cámara no disponible.")
        return

    async with client.expression_measurement.stream.connect(options=stream_options) as socket:
        contador = 1

        while time.time() - tiempo_inicio_total < TIEMPO_LIMITE_SEGUNDOS:
            tiempo_inicio_frame = time.time()
            video_filename = f"temp_video_{contador}.mp4"

            # Configurar VideoWriter
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_filename, fourcc, FPS, RESOLUCION)

            # Capturar frames durante 4999 ms
            frames_a_grabar = int((DURACION_VIDEO_MS / 1000.0) * FPS)
            for _ in range(frames_a_grabar):
                ret, frame = cap.read()
                if not ret:
                    errores_lectura += 1
                    break
                frame_resized = cv2.resize(frame, RESOLUCION)
                out.write(frame_resized)
                await asyncio.sleep(1 / FPS)

            out.release()

            # Enviar video a Hume
            try:
                result = await socket.send_file(video_filename)
            except Exception:
                errores_emociones += 1
                await asyncio.sleep(INTERVALO_SEGUNDOS)
                continue

            try:
                result_dict = result.model_dump()
                face_data = result_dict.get('face', {})

                if not face_data or not face_data.get('predictions'):
                    raise ValueError("Datos de 'face' no disponibles.")

                emotions = face_data['predictions'][0]['emotions']
                emociones_dict = {emocion: 0.0 for emocion in EMOCIONES_OBJETIVO}
                for e in emotions:
                    if e['name'] in emociones_dict:
                        emociones_dict[e['name']] = e['score']

                informe_detallado.append({
                    "video": video_filename,
                    "emociones_encontradas": emociones_dict
                })

                fotogramas_enviados += 1

            except Exception:
                errores_emociones += 1

            tiempos_procesamiento.append(time.time() - tiempo_inicio_frame)
            # os.remove(video_filename)  # Descomenta si deseas eliminar los archivos
            contador += 1
            await asyncio.sleep(INTERVALO_SEGUNDOS)

    tiempo_total = time.time() - tiempo_inicio_total
    fps_promedio = fotogramas_enviados / tiempo_total if tiempo_total > 0 else 0

    # Guardar informe
    with open("informe_emociones.json", "w") as f:
        json.dump({
            "resumen": {
                "duracion_segundos": tiempo_total,
                "videos_enviados": fotogramas_enviados,
                "errores_lectura": errores_lectura,
                "errores_emociones": errores_emociones,
                "frecuencia_envio": fps_promedio
            },
            "detalle": informe_detallado
        }, f, indent=4)

    print(informe_detallado)
    return informe_detallado

    cap.release()
    cv2.destroyAllWindows()

async def main():
    client = AsyncHumeClient(api_key="YOUR_API_KEY")
    await process_video(client)

if __name__ == "__main__":
    asyncio.run(main())
    
