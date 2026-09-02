import os

# Definimos las carpetas que queremos leer y el nombre del archivo final
carpetas = ["svor", "svorex", "svorim"]
archivo_salida = "resultado_combinado.txt"

# Abrimos el archivo de salida en modo escritura ('w') con codificación utf-8
with open(archivo_salida, "w", encoding="utf-8") as f_salida:
    for carpeta in carpetas:
        # Comprobamos si la carpeta realmente existe para evitar errores
        if not os.path.exists(carpeta):
            print(
                f"⚠️ Advertencia: La carpeta '{carpeta}' no existe o no se encuentra. Se omitirá."
            )
            continue

        # Listamos todos los elementos dentro de la carpeta
        for nombre_archivo in os.listdir(carpeta):
            ruta_archivo = os.path.join(carpeta, nombre_archivo)

            # Nos aseguramos de que es un archivo y no una subcarpeta
            if os.path.isfile(ruta_archivo):
                try:
                    # Leemos el contenido de cada archivo
                    with open(ruta_archivo, "r", encoding="utf-8") as f_entrada:
                        contenido = f_entrada.read()

                        # Escribimos un encabezado para identificar el origen del texto
                        f_salida.write(f"\n{'=' * 60}\n")
                        f_salida.write(
                            f"CARPETA: {carpeta} | ARCHIVO: {nombre_archivo}\n"
                        )
                        f_salida.write(f"{'=' * 60}\n\n")

                        # Volcamos el contenido y un par de saltos de línea al final
                        f_salida.write(contenido)
                        f_salida.write("\n\n")

                except UnicodeDecodeError:
                    print(
                        f"❌ Error de codificación: No se pudo leer '{ruta_archivo}'. Asegúrate de que sea un archivo de texto."
                    )
                except Exception as e:
                    print(f"❌ Error inesperado al leer '{ruta_archivo}': {e}")

print(
    f"✅ Proceso completado. El archivo '{archivo_salida}' ha sido generado con éxito."
)
