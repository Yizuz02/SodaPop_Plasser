from flask import Flask, request, jsonify
import pandas as pd
import io

# Importa la función de tu script 'api.py'
from api import get_tamping_predictions, _load_resources

app = Flask(__name__)

# Intenta cargar los modelos al iniciar la aplicación.
# Esto es CRUCIAL para el rendimiento, se hace una sola vez.
if not _load_resources():
    # Si la carga falla (e.g., archivos no encontrados), el servidor puede no iniciar
    # o el endpoint devolverá un error.
    print(
        "¡ADVERTENCIA! La carga de modelos falló al inicio. El endpoint devolverá un error."
    )


@app.route("/predict", methods=["POST"])
def predict():
    """
    Endpoint para recibir un archivo CSV y devolver las predicciones de ajuste.

    Espera un archivo llamado 'data_file' en la petición multipart/form-data.
    """

    # 1. Verificar el tipo de petición y la existencia del archivo
    if "data_file" not in request.files:
        return (
            jsonify({"error": "No se encontró el archivo 'data_file' en la petición."}),
            400,
        )

    data_file = request.files["data_file"]

    if data_file.filename == "":
        return jsonify({"error": "El archivo 'data_file' está vacío."}), 400

    if not data_file.filename.endswith(".csv"):
        return jsonify({"error": "El archivo debe ser de formato CSV."}), 400

    try:
        # 2. Leer el contenido del archivo subido a un buffer
        # Usamos io.StringIO para leer el archivo subido directamente a un DataFrame
        stream = io.StringIO(data_file.stream.read().decode("UTF8"))
        df_crudo = pd.read_csv(stream)

        # 3. Validar estructura básica (opcional, pero recomendado)
        required_cols = ["Time"]  # Puedes agregar más si es necesario
        if not all(col in df_crudo.columns for col in required_cols):
            return (
                jsonify(
                    {
                        "error": f"Faltan columnas requeridas en el CSV. Se requiere al menos: {', '.join(required_cols)}."
                    }
                ),
                400,
            )

        # 4. Ejecutar la función de predicción principal
        predictions = get_tamping_predictions(df_crudo)

        # 5. Devolver el resultado
        if isinstance(predictions, list):
            # Éxito: retorna la lista de predicciones JSON
            return jsonify(predictions), 200
        else:
            # Fallo: retorna el diccionario de error generado en api.py
            return jsonify(predictions), 500

    except Exception as e:
        # Manejar cualquier excepción inesperada durante el procesamiento
        return (
            jsonify(
                {
                    "error": f"Error interno del servidor durante el procesamiento: {str(e)}"
                }
            ),
            500,
        )


if __name__ == "__main__":
    # Ejecutar la aplicación en modo debug para desarrollo
    app.run(host="0.0.0.0", port=5000, debug=True)
