import cv2
import numpy as np


def main():
    # Iniciar la captura de video desde la cámara
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se pudo abrir la cámara.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo leer el frame de la cámara.")
            break

        # Redimensionar el frame si es necesario (opcional)
        scale_percent = 100  # Cambia el porcentaje si quieres redimensionar
        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

        # Convertir a escala de grises y aplicar desenfoque
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blurred = cv2.medianBlur(gray, 5)

        # Detectar círculos
        circles = cv2.HoughCircles(
            gray_blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=100,
            param1=50,
            param2=30,
            minRadius=50,
            maxRadius=200,
        )

        # Dibujar los círculos detectados
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 2)
                cv2.circle(frame, (i[0], i[1]), 2, (0, 0, 255), 3)

        # Mostrar el frame resultante
        cv2.imshow("Detección de Reloj en Tiempo Real", frame)

        # Salir al presionar la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Liberar la cámara y cerrar ventanas
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
