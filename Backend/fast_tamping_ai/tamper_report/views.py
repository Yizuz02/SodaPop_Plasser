import time
import cv2
from django.contrib.auth.models import User
from django.http import HttpResponse, HttpResponseNotFound, StreamingHttpResponse
from django.shortcuts import get_object_or_404
from .models import (
    Station,
    Train,
    TrainTrip,
    TamperMachine,
    TamperOperation,
    Report,
    ReportBatch,
    Route,
)
from rest_framework import viewsets, permissions
from .serializers import (
    ReportUpdateSerializer,
    TamperMachineStatusSerializer,
    UserSerializer,
    StationSerializer,
    TrainSerializer,
    TamperOperationSerializer,
    TamperMachineSerializer,
)
from .serializers import (
    ReportBatchCreateSerializer,
    TrainTripSerializer,
    ReportBatchSerializer,
    ReportSerializer,
    RouteSerializer,
)
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import os
from django.conf import settings
from .tasks import detectar_dormideros_cv, final_tamping_predictions

CROP_WIDTH = 320
CROP_HEIGHT = 240


class UserRoleAPI(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        user = request.user

        return Response(
            {
                "id": user.id,
                "username": user.username,
                "is_staff": user.is_staff,
                "is_superuser": user.is_superuser,
                "groups": [g.name for g in user.groups.all()],
            }
        )


class UserViewSet(viewsets.ModelViewSet):
    queryset = User.objects.all()
    serializer_class = UserSerializer
    permission_classes = [permissions.IsAdminUser]


class StationViewSet(viewsets.ModelViewSet):
    queryset = Station.objects.all()
    serializer_class = StationSerializer

    def get_permissions(self):
        if self.action in ["create", "update", "partial_update", "destroy"]:
            return [permissions.IsAdminUser()]

        return [permissions.IsAuthenticated()]


class TrainViewSet(viewsets.ModelViewSet):
    queryset = Train.objects.all()
    serializer_class = TrainSerializer

    def get_permissions(self):
        if self.action in ["create", "update", "partial_update", "destroy"]:
            return [permissions.IsAdminUser()]

        return [permissions.IsAuthenticated()]


class TrainTripViewSet(viewsets.ModelViewSet):
    queryset = TrainTrip.objects.all()
    serializer_class = TrainTripSerializer

    def get_permissions(self):
        if self.action in ["create", "update", "partial_update", "destroy"]:
            return [permissions.IsAdminUser()]

        return [permissions.IsAuthenticated()]


class RouteViewSet(viewsets.ModelViewSet):
    queryset = Route.objects.all()
    serializer_class = RouteSerializer

    def get_permissions(self):
        if self.action in ["create", "update", "partial_update", "destroy"]:
            return [permissions.IsAdminUser()]

        return [permissions.IsAuthenticated()]


class TrainTripViewSet(viewsets.ModelViewSet):
    queryset = TrainTrip.objects.all()
    serializer_class = TrainTripSerializer

    def get_permissions(self):
        if self.action in ["create", "update", "partial_update", "destroy"]:
            return [permissions.IsAdminUser()]

        return [permissions.IsAuthenticated()]


class TamperMachineViewSet(viewsets.ModelViewSet):
    queryset = TamperMachine.objects.all()
    serializer_class = TamperMachineSerializer

    def get_permissions(self):
        if self.action in ["create", "update", "partial_update", "destroy"]:
            return [permissions.IsAdminUser()]

        return [permissions.IsAuthenticated()]


class TamperOperationViewSet(viewsets.ModelViewSet):
    queryset = TamperOperation.objects.all()
    serializer_class = TamperOperationSerializer

    def get_permissions(self):
        if self.action == "destroy":
            return [permissions.IsAdminUser()]

        return [permissions.IsAuthenticated()]


class ReportBatchAPI(APIView):

    def get_permissions(self):
        if self.request.method == "POST":
            return [permissions.IsAdminUser()]  # Solo admins pueden hacer POST
        elif self.request.method == "GET":
            return [permissions.IsAuthenticated()]  # Cualquier usuario autenticado
        return super().get_permissions()

    def post(self, request):
        serializer = ReportBatchCreateSerializer(data=request.data)
        if serializer.is_valid():
            batch = serializer.save()
            return Response(
                {"message": "Report batch created successfully", "batch_id": batch.id},
                status=status.HTTP_201_CREATED,
            )

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def get(self, request):
        batch_id = request.query_params.get("id")

        if batch_id:
            batch = get_object_or_404(ReportBatch, id=batch_id)
            serializer = ReportBatchSerializer(batch)
            return Response(serializer.data, status=status.HTTP_200_OK)

        batches = ReportBatch.objects.all()
        serializer = ReportBatchSerializer(batches, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)


class ReportUpdateAPI(APIView):
    permission_classes = [permissions.IsAdminUser]  # opcional

    def patch(self, request, report_id):
        report = get_object_or_404(Report, id=report_id)

        serializer = ReportUpdateSerializer(
            report,
            data=request.data,
            partial=True,  # permite actualizar uno o ambos campos
        )

        if serializer.is_valid():
            serializer.save()
            return Response(
                {"message": "Report updated successfully", "report": serializer.data},
                status=status.HTTP_200_OK,
            )

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    # Permitir también PUT si quieres
    def put(self, request, report_id):
        return self.patch(request, report_id)


class TamperMachineStatusAPI(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        machines = TamperMachine.objects.all()
        serializer = TamperMachineStatusSerializer(machines, many=True)
        return Response(serializer.data)


def is_machine_busy(machine_id):
    try:
        machine = TamperMachine.objects.get(id=machine_id)
    except TamperMachine.DoesNotExist:
        return None  # máquina no existe

    ongoing_op = machine.tamperoperation_set.filter(end_time__isnull=True).first()
    return ongoing_op is not None


def generate_frames(video_path, machine_id):
    while True:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"No se pudo abrir el video: {video_path}")
            break

        delay = 1 / 30
        success, frame = cap.read()
        while cap.isOpened():
            busy = is_machine_busy(machine_id)
            if not success:
                break  # fin del video, reinicia

            new_width = CROP_WIDTH
            h, w = frame.shape[:2]

            # calcular altura proporcional
            aspect_ratio = new_width / w
            new_height = int(h * aspect_ratio)

            frame = cv2.resize(frame, (new_width, new_height))

            # Crop central
            h, w, _ = frame.shape
            x_start = max((w - CROP_WIDTH) // 2, 0)
            y_start = max((h - CROP_HEIGHT) // 2, 0)
            frame_cropped = frame[
                y_start : y_start + CROP_HEIGHT, x_start : x_start + CROP_WIDTH
            ]
            if busy:
                frame_cropped = detectar_dormideros_cv(frame_cropped)
            # Convierte frame a JPEG
            _, buffer = cv2.imencode(".jpg", frame_cropped)
            frame_bytes = buffer.tobytes()

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )

            time.sleep(delay)

            if busy:
                success, frame = cap.read()

        cap.release()


def stream_video_loop(request, machine_id):

    busy = is_machine_busy(machine_id)
    # ❌ Máquina no existe
    if busy is None:
        return HttpResponseNotFound("Machine not found")

    video_path = os.path.join(settings.MEDIA_ROOT, f"videos/video{machine_id}.mp4")

    return StreamingHttpResponse(
        generate_frames(video_path, machine_id),
        content_type="multipart/x-mixed-replace; boundary=frame",
    )


class TamperReportAPIView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        """
        Ejecuta el modelo, obtiene el JSON
        y lo guarda con el serializer.
        """

        # 1. Ejecutar tu modelo
        inference_path = os.path.join(settings.DATASET_ROOT, "datos_input.csv")
        json_data = final_tamping_predictions(inference_path)
        print("Predictions obtained:", json_data)

        # 2. Inyectar resultado en el serializer
        data_to_save = {
            "train_trip": 1,
            "reports": json_data,
        }
        print("Data to save:", data_to_save)
        serializer = ReportBatchCreateSerializer(data=data_to_save)

        # 3. Guardar si es válido
        if serializer.is_valid():
            batch = serializer.save()
            return Response(
                {
                    "message": "Reporte procesado y guardado",
                    "batch_id": batch.id,
                },
                status=status.HTTP_201_CREATED,
            )

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
