"""
Railroad Simulator API - Enhanced Version
Generates realistic simulation videos for:
1. Top-down track camera views (3 videos with different terrain types)
2. Tamping machine views (2 videos with side and underside perspectives)
"""

from flask import Flask, Response, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import io
import tempfile
import os
import subprocess
from dataclasses import dataclass
from typing import Optional, Tuple, List
import math

app = Flask(__name__)
CORS(app)

# Configuration - Closer view
VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720
FPS = 30

# Video cache to avoid regenerating videos
VIDEO_CACHE = {}
CACHE_ENABLED = True


@dataclass
class VideoConfig:
    duration: float
    speed: float = 1.0
    terrain_type: str = "gravel"  # gravel, desert, forest


@dataclass
class TampingConfig:
    duration: float
    stop_time: float
    stop_duration: float
    speed: float = 1.0
    terrain_type: str = "industrial"


class RealisticTrackRenderer:
    """Renders highly realistic railroad tracks from top-down perspective - CLOSER VIEW"""

    # Terrain color palettes
    TERRAINS = {
        "gravel": {
            "base": (75, 70, 65),
            "gravel_colors": [
                (90, 85, 78),
                (70, 65, 60),
                (100, 95, 88),
                (60, 58, 55),
                (80, 75, 70),
            ],
            "dirt_patches": [(65, 55, 45), (55, 48, 40)],
            "vegetation": None,
        },
        "desert": {
            "base": (120, 140, 180),
            "gravel_colors": [
                (130, 150, 190),
                (110, 130, 170),
                (140, 155, 185),
                (100, 120, 160),
            ],
            "dirt_patches": [(100, 115, 145), (90, 105, 135)],
            "vegetation": [(80, 140, 100), (70, 120, 85)],
        },
        "forest": {
            "base": (50, 65, 45),
            "gravel_colors": [(65, 75, 55), (55, 70, 50), (70, 80, 60), (45, 60, 40)],
            "dirt_patches": [(45, 40, 35), (55, 50, 42)],
            "vegetation": [(60, 100, 50), (50, 90, 40), (70, 110, 55)],
        },
    }

    def __init__(self, width: int, height: int, terrain_type: str = "gravel"):
        self.width = width
        self.height = height
        self.terrain = self.TERRAINS.get(terrain_type, self.TERRAINS["gravel"])
        self.terrain_type = terrain_type

        # CLOSER VIEW - Larger elements
        self.rail_spacing = 280  # Wider rail gauge (closer view)
        self.rail_width = 28  # Thicker rails
        self.tie_spacing = 90  # Spacing between ties
        self.tie_width = 420  # Wider ties
        self.tie_height = 38  # Taller ties

        # Pre-generate static noise texture for performance
        self.noise_texture = self._generate_noise_texture()

    def _generate_noise_texture(self) -> np.ndarray:
        """Generate a reusable noise texture"""
        noise = np.random.randint(0, 30, (self.height, self.width), dtype=np.uint8)
        return noise

    def draw_terrain_base(self, frame: np.ndarray, offset: float) -> None:
        """Draw realistic terrain base with texture"""
        # Fill with base color
        frame[:] = self.terrain["base"]

        # Add noise texture for realism
        noise_offset = int(offset) % 100
        noise = np.roll(self.noise_texture, noise_offset, axis=0)

        for c in range(3):
            frame[:, :, c] = np.clip(
                frame[:, :, c].astype(np.int16) + (noise.astype(np.int16) - 15), 0, 255
            ).astype(np.uint8)

    def draw_ballast_bed(self, frame: np.ndarray, offset: float) -> None:
        """Draw the gravel/ballast bed under and around tracks"""
        center_x = self.width // 2
        ballast_width = self.tie_width + 100

        # Ballast bed shape (trapezoidal cross-section visible from top)
        for y in range(self.height):
            x1 = center_x - ballast_width // 2
            x2 = center_x + ballast_width // 2

            # Draw ballast strip
            cv2.line(frame, (x1, y), (x2, y), self.terrain["gravel_colors"][0], 1)

        # Add individual stones/gravel pieces
        np.random.seed(int(offset * 5) % 10000)
        num_stones = 2000

        for _ in range(num_stones):
            x = np.random.randint(
                center_x - ballast_width // 2, center_x + ballast_width // 2
            )
            y = np.random.randint(0, self.height)

            # Vary stone sizes - LARGER for closer view
            size = np.random.choice(
                [3, 4, 5, 6, 7, 8], p=[0.15, 0.25, 0.25, 0.2, 0.1, 0.05]
            )

            # Random color from palette
            color = self.terrain["gravel_colors"][
                np.random.randint(0, len(self.terrain["gravel_colors"]))
            ]

            # Add slight color variation
            color = tuple(
                max(0, min(255, c + np.random.randint(-15, 15))) for c in color
            )

            # Draw stone (slightly irregular)
            if np.random.random() > 0.3:
                cv2.circle(frame, (x, y), size, color, -1)
            else:
                # Elongated stone
                angle = np.random.randint(0, 180)
                axes = (size + np.random.randint(0, 4), size)
                cv2.ellipse(frame, (x, y), axes, angle, 0, 360, color, -1)

    def draw_ties(self, frame: np.ndarray, offset: float) -> None:
        """Draw realistic wooden railroad ties with wood grain texture"""
        start_y = int(-offset % self.tie_spacing) - self.tie_height
        center_x = self.width // 2

        for y in range(start_y, self.height + self.tie_height, self.tie_spacing):
            x1 = center_x - self.tie_width // 2

            # Tie base colors (weathered wood)
            tie_colors = [
                (55, 45, 38),  # Dark wood
                (65, 52, 42),  # Medium wood
                (50, 42, 35),  # Very dark
                (70, 58, 48),  # Lighter wood
            ]
            base_color = tie_colors[int(offset + y) % len(tie_colors)]

            # Draw main tie body
            cv2.rectangle(
                frame,
                (x1, y),
                (x1 + self.tie_width, y + self.tie_height),
                base_color,
                -1,
            )

            # Wood grain lines
            np.random.seed(int(y * 100 + offset) % 10000)
            num_grain_lines = 12
            for i in range(num_grain_lines):
                grain_y = y + 3 + (i * (self.tie_height - 6) // num_grain_lines)
                grain_y += np.random.randint(-1, 2)

                shade = np.random.randint(-20, 10)
                grain_color = tuple(max(0, min(255, c + shade)) for c in base_color)

                # Grain line with slight waviness
                thickness = np.random.choice([1, 1, 1, 2])
                cv2.line(
                    frame,
                    (x1 + 8, grain_y),
                    (x1 + self.tie_width - 8, grain_y),
                    grain_color,
                    thickness,
                )

            # Add wood knots occasionally
            if np.random.random() > 0.6:
                knot_x = x1 + np.random.randint(30, self.tie_width - 30)
                knot_y = y + np.random.randint(8, self.tie_height - 8)
                knot_color = (40, 32, 28)
                cv2.circle(
                    frame, (knot_x, knot_y), np.random.randint(4, 8), knot_color, -1
                )
                cv2.circle(
                    frame, (knot_x, knot_y), np.random.randint(2, 5), (35, 28, 24), -1
                )

            # Weathering/damage marks
            if np.random.random() > 0.7:
                crack_x = x1 + np.random.randint(20, self.tie_width - 20)
                crack_y1 = y + 2
                crack_y2 = y + np.random.randint(10, self.tie_height - 2)
                cv2.line(
                    frame,
                    (crack_x, crack_y1),
                    (crack_x + np.random.randint(-5, 5), crack_y2),
                    (30, 25, 20),
                    1,
                )

            # Tie plate shadows (where rails sit)
            for side in [-1, 1]:
                plate_x = center_x + side * self.rail_spacing // 2 - 25
                plate_y = y + 2
                cv2.rectangle(
                    frame,
                    (plate_x, plate_y),
                    (plate_x + 50, y + self.tie_height - 2),
                    (45, 38, 32),
                    -1,
                )

    def draw_rails(self, frame: np.ndarray, offset: float) -> None:
        """Draw realistic steel rails with reflections and wear patterns"""
        center_x = self.width // 2

        for side in [-1, 1]:
            rail_center_x = center_x + side * self.rail_spacing // 2

            # Rail shadow (underneath)
            shadow_offset = 4
            cv2.rectangle(
                frame,
                (rail_center_x - self.rail_width // 2 + shadow_offset, 0),
                (rail_center_x + self.rail_width // 2 + shadow_offset, self.height),
                (30, 28, 25),
                -1,
            )

            # Rail base (dark steel)
            cv2.rectangle(
                frame,
                (rail_center_x - self.rail_width // 2, 0),
                (rail_center_x + self.rail_width // 2, self.height),
                (85, 82, 78),
                -1,
            )

            # Rail web (middle section - slightly darker)
            web_width = self.rail_width // 3
            cv2.rectangle(
                frame,
                (rail_center_x - web_width // 2, 0),
                (rail_center_x + web_width // 2, self.height),
                (75, 72, 68),
                -1,
            )

            # Rail head (top running surface - shiny/worn)
            head_width = self.rail_width * 2 // 3
            cv2.rectangle(
                frame,
                (rail_center_x - head_width // 2, 0),
                (rail_center_x + head_width // 2, self.height),
                (140, 138, 135),
                -1,
            )

            # Shiny center strip (wheel contact area)
            shine_width = head_width // 3
            cv2.rectangle(
                frame,
                (rail_center_x - shine_width // 2, 0),
                (rail_center_x + shine_width // 2, self.height),
                (180, 178, 175),
                -1,
            )

            # Add occasional rust spots and wear marks
            np.random.seed(int(offset * 3 + side * 1000) % 10000)
            for _ in range(15):
                spot_y = np.random.randint(0, self.height)
                spot_x = rail_center_x + np.random.randint(
                    -self.rail_width // 3, self.rail_width // 3
                )
                spot_size = np.random.randint(2, 5)

                # Rust color
                rust_color = (55, 75, 95 + np.random.randint(0, 30))
                cv2.circle(frame, (spot_x, spot_y), spot_size, rust_color, -1)

    def draw_rail_fasteners(self, frame: np.ndarray, offset: float) -> None:
        """Draw rail clips/fasteners at tie positions"""
        start_y = int(-offset % self.tie_spacing) - self.tie_height
        center_x = self.width // 2

        for y in range(start_y, self.height + self.tie_height, self.tie_spacing):
            fastener_y = y + self.tie_height // 2

            for side in [-1, 1]:
                rail_x = center_x + side * self.rail_spacing // 2

                # Draw fastener clips on both sides of rail
                for clip_side in [-1, 1]:
                    clip_x = rail_x + clip_side * (self.rail_width // 2 + 8)

                    # Fastener clip
                    cv2.rectangle(
                        frame,
                        (clip_x - 6, fastener_y - 5),
                        (clip_x + 6, fastener_y + 5),
                        (70, 68, 65),
                        -1,
                    )

                    # Bolt head
                    cv2.circle(frame, (clip_x, fastener_y), 4, (90, 88, 85), -1)
                    cv2.circle(frame, (clip_x, fastener_y), 2, (60, 58, 55), -1)

    def draw_environment_details(self, frame: np.ndarray, offset: float) -> None:
        """Add environmental details based on terrain type"""
        center_x = self.width // 2
        edge_zone = self.tie_width // 2 + 80

        np.random.seed(int(offset * 7) % 10000)

        if self.terrain.get("vegetation"):
            # Add grass/vegetation at edges
            for _ in range(100):
                side = np.random.choice([-1, 1])
                x = center_x + side * (edge_zone + np.random.randint(10, 100))
                y = np.random.randint(0, self.height)

                if 0 <= x < self.width:
                    color = self.terrain["vegetation"][
                        np.random.randint(0, len(self.terrain["vegetation"]))
                    ]

                    # Draw grass blade or small plant
                    blade_height = np.random.randint(8, 20)
                    sway = np.random.randint(-3, 4)
                    cv2.line(frame, (x, y), (x + sway, y - blade_height), color, 2)

        # Add dirt patches
        if self.terrain.get("dirt_patches"):
            for _ in range(30):
                x = np.random.randint(0, self.width)
                y = np.random.randint(0, self.height)

                # Only draw outside the ballast area
                if abs(x - center_x) > edge_zone - 50:
                    color = self.terrain["dirt_patches"][
                        np.random.randint(0, len(self.terrain["dirt_patches"]))
                    ]
                    size = np.random.randint(10, 30)
                    cv2.circle(frame, (x, y), size, color, -1)

    def render_frame(self, offset: float) -> np.ndarray:
        """Render a complete frame"""
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        self.draw_terrain_base(frame, offset)
        self.draw_environment_details(frame, offset)
        self.draw_ballast_bed(frame, offset)
        self.draw_ties(frame, offset)
        self.draw_rails(frame, offset)
        self.draw_rail_fasteners(frame, offset)

        return frame


class RealisticTampingRenderer:
    """Renders realistic tamping machine simulation"""

    TERRAINS = {
        "industrial": {
            "sky_top": (180, 140, 120),
            "sky_bottom": (220, 200, 180),
            "ground": (75, 70, 65),
        },
        "rural": {
            "sky_top": (200, 160, 140),
            "sky_bottom": (235, 220, 200),
            "ground": (60, 80, 55),
        },
    }

    def __init__(self, width: int, height: int, terrain_type: str = "industrial"):
        self.width = width
        self.height = height
        self.terrain = self.TERRAINS.get(terrain_type, self.TERRAINS["industrial"])

        # Machine colors (yellow/orange construction equipment)
        self.machine_body = (0, 140, 220)  # Yellow in BGR
        self.machine_dark = (0, 100, 180)
        self.machine_accent = (30, 90, 160)
        self.cabin_color = (40, 40, 45)
        self.window_color = (180, 160, 140)

    def draw_sky_gradient(self, frame: np.ndarray) -> None:
        """Draw realistic sky gradient"""
        horizon_y = self.height // 2 + 50

        for y in range(horizon_y):
            ratio = y / horizon_y
            color = tuple(
                int(
                    self.terrain["sky_top"][c] * (1 - ratio)
                    + self.terrain["sky_bottom"][c] * ratio
                )
                for c in range(3)
            )
            cv2.line(frame, (0, y), (self.width, y), color, 1)

    def draw_ground(self, frame: np.ndarray, offset: float) -> None:
        """Draw ground with texture"""
        horizon_y = self.height // 2 + 50

        # Base ground color
        cv2.rectangle(
            frame, (0, horizon_y), (self.width, self.height), self.terrain["ground"], -1
        )

        # Add texture
        np.random.seed(int(offset * 3) % 1000)
        for _ in range(500):
            x = np.random.randint(0, self.width)
            y = np.random.randint(horizon_y, self.height)
            size = np.random.randint(2, 6)
            shade = np.random.randint(-20, 20)
            color = tuple(
                max(0, min(255, self.terrain["ground"][c] + shade)) for c in range(3)
            )
            cv2.circle(frame, (x, y), size, color, -1)

    def draw_tracks_side_view(self, frame: np.ndarray, offset: float) -> None:
        """Draw tracks from side view with perspective"""
        track_y = self.height // 2 + 70

        # Ballast bed
        ballast_pts = np.array(
            [
                [0, track_y + 40],
                [0, track_y + 15],
                [self.width, track_y + 15],
                [self.width, track_y + 40],
            ],
            np.int32,
        )
        cv2.fillPoly(frame, [ballast_pts], (90, 85, 80))

        # Rail (side profile)
        rail_height = 25
        # Rail base
        cv2.rectangle(
            frame, (0, track_y), (self.width, track_y + rail_height), (85, 82, 78), -1
        )
        # Rail head (shiny top)
        cv2.rectangle(
            frame, (0, track_y), (self.width, track_y + 8), (150, 148, 145), -1
        )
        # Highlight
        cv2.rectangle(
            frame, (0, track_y + 2), (self.width, track_y + 5), (180, 178, 175), -1
        )

        # Ties (visible from side)
        tie_spacing = 70
        start_x = int(-offset * 80) % tie_spacing - 20

        for x in range(start_x, self.width + 20, tie_spacing):
            # Tie front face
            cv2.rectangle(
                frame,
                (x, track_y + rail_height),
                (x + 18, track_y + 55),
                (55, 45, 38),
                -1,
            )
            # Tie shadow
            cv2.rectangle(
                frame,
                (x + 18, track_y + rail_height + 5),
                (x + 22, track_y + 55),
                (40, 32, 28),
                -1,
            )

    def draw_tamping_machine(
        self, frame: np.ndarray, x_pos: int, working: bool, frame_num: int
    ) -> None:
        """Draw detailed tamping machine from side view"""
        base_y = self.height // 2 - 20
        machine_width = 380
        machine_height = 140

        # Main body shadow
        shadow_offset = 8
        body_shadow = np.array(
            [
                [x_pos + shadow_offset, base_y + machine_height],
                [x_pos + 30 + shadow_offset, base_y + 30],
                [x_pos + machine_width - 30 + shadow_offset, base_y + 30],
                [x_pos + machine_width + shadow_offset, base_y + machine_height],
            ],
            np.int32,
        )
        cv2.fillPoly(frame, [body_shadow], (40, 35, 30))

        # Main body
        body_pts = np.array(
            [
                [x_pos, base_y + machine_height],
                [x_pos + 30, base_y + 30],
                [x_pos + machine_width - 30, base_y + 30],
                [x_pos + machine_width, base_y + machine_height],
            ],
            np.int32,
        )
        cv2.fillPoly(frame, [body_pts], self.machine_body)

        # Body details/panels
        cv2.line(
            frame,
            (x_pos + 50, base_y + 40),
            (x_pos + 50, base_y + machine_height - 10),
            self.machine_dark,
            3,
        )
        cv2.line(
            frame,
            (x_pos + machine_width - 50, base_y + 40),
            (x_pos + machine_width - 50, base_y + machine_height - 10),
            self.machine_dark,
            3,
        )

        # Vents/grilles
        for i in range(4):
            vent_x = x_pos + 70 + i * 25
            cv2.rectangle(
                frame,
                (vent_x, base_y + 50),
                (vent_x + 15, base_y + 80),
                self.machine_dark,
                -1,
            )

        # Cabin
        cabin_x = x_pos + 60
        cabin_y = base_y - 50
        cabin_w = 120
        cabin_h = 85

        # Cabin body
        cv2.rectangle(
            frame,
            (cabin_x, cabin_y),
            (cabin_x + cabin_w, cabin_y + cabin_h),
            self.cabin_color,
            -1,
        )

        # Windows
        cv2.rectangle(
            frame,
            (cabin_x + 10, cabin_y + 10),
            (cabin_x + cabin_w - 10, cabin_y + 50),
            self.window_color,
            -1,
        )
        # Window frame
        cv2.rectangle(
            frame,
            (cabin_x + 10, cabin_y + 10),
            (cabin_x + cabin_w - 10, cabin_y + 50),
            (30, 30, 35),
            2,
        )
        # Window divider
        cv2.line(
            frame,
            (cabin_x + cabin_w // 2, cabin_y + 10),
            (cabin_x + cabin_w // 2, cabin_y + 50),
            (30, 30, 35),
            2,
        )

        # Roof
        roof_pts = np.array(
            [
                [cabin_x - 5, cabin_y],
                [cabin_x + 10, cabin_y - 15],
                [cabin_x + cabin_w - 10, cabin_y - 15],
                [cabin_x + cabin_w + 5, cabin_y],
            ],
            np.int32,
        )
        cv2.fillPoly(frame, [roof_pts], (50, 50, 55))

        # Wheels (bogies)
        wheel_y = base_y + machine_height + 15
        wheel_positions = [
            x_pos + 55,
            x_pos + 110,
            x_pos + machine_width - 110,
            x_pos + machine_width - 55,
        ]

        for wx in wheel_positions:
            # Wheel housing
            cv2.rectangle(
                frame,
                (wx - 25, base_y + machine_height - 10),
                (wx + 25, wheel_y + 5),
                (50, 50, 55),
                -1,
            )

            # Wheel
            cv2.circle(frame, (wx, wheel_y), 28, (45, 45, 48), -1)
            cv2.circle(frame, (wx, wheel_y), 22, (60, 60, 65), -1)
            cv2.circle(frame, (wx, wheel_y), 12, (50, 50, 55), -1)
            cv2.circle(frame, (wx, wheel_y), 5, (40, 40, 45), -1)

            # Wheel spokes
            for angle in range(0, 360, 45):
                rad = math.radians(angle + frame_num * 3)
                x1 = int(wx + 8 * math.cos(rad))
                y1 = int(wheel_y + 8 * math.sin(rad))
                x2 = int(wx + 20 * math.cos(rad))
                y2 = int(wheel_y + 20 * math.sin(rad))
                cv2.line(frame, (x1, y1), (x2, y2), (70, 70, 75), 2)

        # Tamping unit
        tamp_x = x_pos + machine_width // 2 - 40
        tamp_base_y = base_y + machine_height

        # Tamping frame
        cv2.rectangle(
            frame,
            (tamp_x, tamp_base_y - 20),
            (tamp_x + 80, tamp_base_y + 10),
            (70, 70, 75),
            -1,
        )

        if working:
            # Extended tamping tools with vibration
            vibration = int(math.sin(frame_num * 0.8) * 4)

            for i, tx in enumerate([tamp_x + 10, tamp_x + 35, tamp_x + 60]):
                tool_vibration = vibration if i % 2 == 0 else -vibration

                # Hydraulic cylinder
                cv2.rectangle(
                    frame,
                    (tx, tamp_base_y),
                    (tx + 15, tamp_base_y + 50),
                    (80, 80, 85),
                    -1,
                )

                # Tamping tine
                cv2.rectangle(
                    frame,
                    (tx + tool_vibration, tamp_base_y + 45),
                    (tx + 15 + tool_vibration, tamp_base_y + 90),
                    (100, 100, 105),
                    -1,
                )

                # Tine tip
                cv2.rectangle(
                    frame,
                    (tx - 3 + tool_vibration, tamp_base_y + 85),
                    (tx + 18 + tool_vibration, tamp_base_y + 100),
                    (90, 90, 95),
                    -1,
                )

            # Warning lights (flashing)
            if frame_num % 10 < 5:
                cv2.circle(frame, (x_pos + 25, base_y + 50), 12, (0, 180, 255), -1)
                cv2.circle(
                    frame,
                    (x_pos + machine_width - 25, base_y + 50),
                    12,
                    (0, 180, 255),
                    -1,
                )
                # Glow effect
                cv2.circle(frame, (x_pos + 25, base_y + 50), 18, (0, 100, 180), 2)
                cv2.circle(
                    frame,
                    (x_pos + machine_width - 25, base_y + 50),
                    18,
                    (0, 100, 180),
                    2,
                )

        # Company markings
        cv2.putText(
            frame,
            "PLASSER",
            (x_pos + 200, base_y + 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            "09-4X",
            (x_pos + 200, base_y + 115),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

    def draw_underside_view(
        self, frame: np.ndarray, working: bool, frame_num: int
    ) -> None:
        """Draw dramatic view from underneath looking up at tracks and machine"""
        # Dark atmospheric background
        frame[:] = (25, 22, 18)

        # Add some ambient lighting variation
        for y in range(self.height):
            brightness = int(10 * (1 - y / self.height))
            frame[y, :] = np.clip(
                frame[y, :].astype(np.int16) + brightness, 0, 255
            ).astype(np.uint8)

        center_x = self.width // 2

        # Rails from below with perspective
        rail_spacing_bottom = 350
        rail_spacing_top = 180

        for side in [-1, 1]:
            # Calculate perspective points
            bottom_x = center_x + side * rail_spacing_bottom // 2
            top_x = center_x + side * rail_spacing_top // 2

            # Rail bottom surface (perspective trapezoid)
            rail_pts = np.array(
                [
                    [bottom_x - 30, self.height],
                    [top_x - 15, 0],
                    [top_x + 15, 0],
                    [bottom_x + 30, self.height],
                ],
                np.int32,
            )
            cv2.fillPoly(frame, [rail_pts], (70, 68, 65))

            # Rail inner edge (shiny)
            inner_pts = np.array(
                [
                    [bottom_x - 15, self.height],
                    [top_x - 8, 0],
                    [top_x + 8, 0],
                    [bottom_x + 15, self.height],
                ],
                np.int32,
            )
            cv2.fillPoly(frame, [inner_pts], (95, 93, 90))

        # Ties from below (perspective)
        num_ties = 12
        for i in range(num_ties):
            y = int(self.height * 0.1 + i * (self.height * 0.85 / num_ties))

            # Perspective scaling
            scale = 0.5 + (y / self.height) * 0.8
            tie_width = int(450 * scale)
            tie_height = int(25 * scale)

            tie_x = center_x - tie_width // 2

            # Tie bottom surface
            cv2.rectangle(
                frame, (tie_x, y), (tie_x + tie_width, y + tie_height), (50, 42, 35), -1
            )

            # Add wood texture lines
            for j in range(3):
                line_y = y + 5 + j * 6
                if line_y < y + tie_height:
                    cv2.line(
                        frame,
                        (tie_x + 10, line_y),
                        (tie_x + tie_width - 10, line_y),
                        (45, 38, 32),
                        1,
                    )

        # Machine undercarriage
        undercarriage_y = 80
        cv2.rectangle(
            frame,
            (center_x - 250, 0),
            (center_x + 250, undercarriage_y),
            (35, 32, 28),
            -1,
        )

        # Hydraulic lines and machinery
        for i in range(5):
            pipe_x = center_x - 180 + i * 90
            cv2.line(
                frame, (pipe_x, 0), (pipe_x, undercarriage_y + 30), (55, 52, 48), 8
            )
            cv2.line(
                frame, (pipe_x, 0), (pipe_x, undercarriage_y + 30), (65, 62, 58), 4
            )

        if working:
            # Tamping tools coming down with vibration
            vibration = int(math.sin(frame_num * 0.8) * 6)

            for i, tx in enumerate([center_x - 80, center_x, center_x + 80]):
                tool_vib = vibration if i % 2 == 0 else -vibration

                # Tool body
                cv2.rectangle(
                    frame, (tx - 20, undercarriage_y), (tx + 20, 350), (85, 82, 78), -1
                )

                # Tool head
                cv2.rectangle(
                    frame,
                    (tx - 30 + tool_vib, 320),
                    (tx + 30 + tool_vib, 400),
                    (100, 98, 95),
                    -1,
                )

                # Tines
                for tine_offset in [-18, 0, 18]:
                    tine_x = tx + tine_offset + tool_vib
                    cv2.rectangle(
                        frame, (tine_x - 5, 390), (tine_x + 5, 480), (90, 88, 85), -1
                    )

            # Flying ballast particles
            np.random.seed(frame_num % 100)
            for _ in range(40):
                px = center_x + np.random.randint(-200, 200)
                py = np.random.randint(350, self.height)
                psize = np.random.randint(3, 8)
                cv2.circle(frame, (px, py), psize, (100, 95, 88), -1)

            # Dust/debris effect
            for _ in range(20):
                dx = center_x + np.random.randint(-150, 150)
                dy = np.random.randint(300, 500)
                cv2.circle(frame, (dx, dy), np.random.randint(1, 4), (80, 75, 70), -1)

    def render_frame_side(
        self, offset: float, machine_x: int, working: bool, frame_num: int
    ) -> np.ndarray:
        """Render side view frame"""
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        self.draw_sky_gradient(frame)
        self.draw_ground(frame, offset)
        self.draw_tracks_side_view(frame, offset)
        self.draw_tamping_machine(frame, machine_x, working, frame_num)

        return frame

    def render_frame_underside(self, working: bool, frame_num: int) -> np.ndarray:
        """Render underside view frame"""
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.draw_underside_view(frame, working, frame_num)
        return frame


def generate_track_video(
    config: VideoConfig, terrain_type: str = "gravel"
) -> io.BytesIO:
    """Generate a top-down track video with specified terrain"""
    renderer = RealisticTrackRenderer(VIDEO_WIDTH, VIDEO_HEIGHT, terrain_type)
    total_frames = int(config.duration * FPS)

    temp_raw = tempfile.NamedTemporaryFile(suffix=".avi", delete=False)
    temp_raw_path = temp_raw.name
    temp_raw.close()

    temp_mp4 = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    temp_mp4_path = temp_mp4.name
    temp_mp4.close()

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out = cv2.VideoWriter(temp_raw_path, fourcc, FPS, (VIDEO_WIDTH, VIDEO_HEIGHT))

    for frame_num in range(total_frames):
        offset = frame_num * 4 * config.speed  # Movement speed
        frame = renderer.render_frame(offset)
        out.write(frame)

    out.release()

    # Convert to H.264
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
                "fast",
                "-crf",
                "23",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                temp_mp4_path,
            ],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e.stderr.decode()}")
        import shutil

        shutil.copy(temp_raw_path, temp_mp4_path)

    with open(temp_mp4_path, "rb") as f:
        video_data = io.BytesIO(f.read())

    os.unlink(temp_raw_path)
    os.unlink(temp_mp4_path)

    return video_data


def generate_tamping_video(config: TampingConfig) -> io.BytesIO:
    """Generate a tamping machine video"""
    renderer = RealisticTampingRenderer(VIDEO_WIDTH, VIDEO_HEIGHT, config.terrain_type)
    total_frames = int(config.duration * FPS)
    stop_start_frame = int(config.stop_time * FPS)
    stop_end_frame = int((config.stop_time + config.stop_duration) * FPS)

    temp_raw = tempfile.NamedTemporaryFile(suffix=".avi", delete=False)
    temp_raw_path = temp_raw.name
    temp_raw.close()

    temp_mp4 = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    temp_mp4_path = temp_mp4.name
    temp_mp4.close()

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out = cv2.VideoWriter(temp_raw_path, fourcc, FPS, (VIDEO_WIDTH, VIDEO_HEIGHT))

    machine_x = -400
    transition_frames = 25

    for frame_num in range(total_frames):
        working = False

        if stop_start_frame <= frame_num < stop_end_frame:
            working = True
            frames_into_stop = frame_num - stop_start_frame
            frames_until_resume = stop_end_frame - frame_num

            if (
                frames_into_stop > transition_frames
                and frames_until_resume > transition_frames
            ):
                frame = renderer.render_frame_underside(
                    working=True, frame_num=frame_num
                )
            else:
                frame = renderer.render_frame_side(0, machine_x, working, frame_num)
        else:
            if frame_num < stop_start_frame:
                progress = frame_num / stop_start_frame if stop_start_frame > 0 else 1
                machine_x = int(-400 + progress * (VIDEO_WIDTH // 2 - 190 + 400))
            else:
                progress = (
                    (frame_num - stop_end_frame) / (total_frames - stop_end_frame)
                    if total_frames > stop_end_frame
                    else 1
                )
                machine_x = int(
                    VIDEO_WIDTH // 2
                    - 190
                    + progress * (VIDEO_WIDTH + 400 - (VIDEO_WIDTH // 2 - 190))
                )

            offset = frame_num * 2 * config.speed
            frame = renderer.render_frame_side(
                offset, machine_x, working=False, frame_num=frame_num
            )

        out.write(frame)

    out.release()

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
                "fast",
                "-crf",
                "23",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                temp_mp4_path,
            ],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e.stderr.decode()}")
        import shutil

        shutil.copy(temp_raw_path, temp_mp4_path)

    with open(temp_mp4_path, "rb") as f:
        video_data = io.BytesIO(f.read())

    os.unlink(temp_raw_path)
    os.unlink(temp_mp4_path)

    return video_data


# API Routes


@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "service": "Railroad Simulator API v2.0"})


@app.route("/api/videos/track/<int:video_id>", methods=["GET"])
def get_track_video(video_id: int):
    """Get track survey video with different terrains"""
    if video_id not in [1, 2, 3]:
        return jsonify({"error": "Invalid video_id. Must be 1, 2, or 3"}), 400

    cache_key = f"track_{video_id}"

    # Check cache first
    if CACHE_ENABLED and cache_key in VIDEO_CACHE:
        print(f"[CACHE HIT] Returning cached video for {cache_key}")
        video_data = VIDEO_CACHE[cache_key]
        video_data.seek(0)
        return Response(
            video_data.read(),
            mimetype="video/mp4",
            headers={
                "Content-Disposition": f"inline; filename=track_survey_{video_id}.mp4",
                "Cache-Control": "public, max-age=3600",
            },
        )

    print(f"[CACHE MISS] Generating video for {cache_key}...")

    # Different terrain for each video
    configs = {
        1: {
            "duration": 10.0,
            "speed": 1.2,
            "terrain": "gravel",
        },  # Standard gravel ballast
        2: {
            "duration": 10.0,
            "speed": 1.0,
            "terrain": "desert",
        },  # Desert/sandy terrain
        3: {
            "duration": 10.0,
            "speed": 1.1,
            "terrain": "forest",
        },  # Forest/rural terrain
    }

    cfg = configs[video_id]
    video_data = generate_track_video(
        VideoConfig(
            duration=cfg["duration"], speed=cfg["speed"], terrain_type=cfg["terrain"]
        ),
        terrain_type=cfg["terrain"],
    )

    # Store in cache
    if CACHE_ENABLED:
        VIDEO_CACHE[cache_key] = video_data
        print(f"[CACHE STORE] Stored video for {cache_key}")

    video_data.seek(0)

    return Response(
        video_data.read(),
        mimetype="video/mp4",
        headers={
            "Content-Disposition": f"inline; filename=track_survey_{video_id}.mp4",
            "Cache-Control": "public, max-age=3600",
        },
    )


@app.route("/api/videos/tamping/<int:machine_id>", methods=["GET"])
def get_tamping_video(machine_id: int):
    """Get tamping machine video"""
    if machine_id not in [1, 2]:
        return jsonify({"error": "Invalid machine_id. Must be 1 or 2"}), 400

    cache_key = f"tamping_{machine_id}"

    # Check cache first
    if CACHE_ENABLED and cache_key in VIDEO_CACHE:
        print(f"[CACHE HIT] Returning cached video for {cache_key}")
        video_data = VIDEO_CACHE[cache_key]
        video_data.seek(0)
        return Response(
            video_data.read(),
            mimetype="video/mp4",
            headers={
                "Content-Disposition": f"inline; filename=tamping_machine_{machine_id}.mp4",
                "Cache-Control": "public, max-age=3600",
            },
        )

    print(f"[CACHE MISS] Generating video for {cache_key}...")

    configs = {
        1: TampingConfig(
            duration=15.0,
            stop_time=4.0,
            stop_duration=5.0,
            speed=1.0,
            terrain_type="industrial",
        ),
        2: TampingConfig(
            duration=15.0,
            stop_time=6.0,
            stop_duration=4.0,
            speed=0.9,
            terrain_type="rural",
        ),
    }

    config = configs[machine_id]
    video_data = generate_tamping_video(config)

    # Store in cache
    if CACHE_ENABLED:
        VIDEO_CACHE[cache_key] = video_data
        print(f"[CACHE STORE] Stored video for {cache_key}")

    video_data.seek(0)

    return Response(
        video_data.read(),
        mimetype="video/mp4",
        headers={
            "Content-Disposition": f"inline; filename=tamping_machine_{machine_id}.mp4",
            "Cache-Control": "public, max-age=3600",
        },
    )


@app.route("/api/videos/info", methods=["GET"])
def get_videos_info():
    return jsonify(
        {
            "track_videos": [
                {
                    "id": 1,
                    "description": "Track survey - Gravel ballast terrain",
                    "duration": 10,
                    "terrain": "gravel",
                },
                {
                    "id": 2,
                    "description": "Track survey - Desert/sandy terrain",
                    "duration": 10,
                    "terrain": "desert",
                },
                {
                    "id": 3,
                    "description": "Track survey - Forest/rural terrain",
                    "duration": 10,
                    "terrain": "forest",
                },
            ],
            "tamping_videos": [
                {
                    "id": 1,
                    "description": "Tamping machine - Industrial area, stops at 4s",
                    "duration": 15,
                },
                {
                    "id": 2,
                    "description": "Tamping machine - Rural area, stops at 6s",
                    "duration": 15,
                },
            ],
            "cache_status": {
                "enabled": CACHE_ENABLED,
                "cached_videos": list(VIDEO_CACHE.keys()),
            },
        }
    )


@app.route("/api/cache/warmup", methods=["POST"])
def warmup_cache():
    """Pre-generate all videos to cache"""
    print("=" * 50)
    print("WARMING UP CACHE - Generating all videos...")
    print("=" * 50)

    results = {}

    # Generate track videos
    for video_id in [1, 2, 3]:
        cache_key = f"track_{video_id}"
        if cache_key not in VIDEO_CACHE:
            print(f"Generating {cache_key}...")
            configs = {
                1: {"duration": 10.0, "speed": 1.2, "terrain": "gravel"},
                2: {"duration": 10.0, "speed": 1.0, "terrain": "desert"},
                3: {"duration": 10.0, "speed": 1.1, "terrain": "forest"},
            }
            cfg = configs[video_id]
            video_data = generate_track_video(
                VideoConfig(
                    duration=cfg["duration"],
                    speed=cfg["speed"],
                    terrain_type=cfg["terrain"],
                ),
                terrain_type=cfg["terrain"],
            )
            VIDEO_CACHE[cache_key] = video_data
            results[cache_key] = "generated"
        else:
            results[cache_key] = "already cached"

    # Generate tamping videos
    for machine_id in [1, 2]:
        cache_key = f"tamping_{machine_id}"
        if cache_key not in VIDEO_CACHE:
            print(f"Generating {cache_key}...")
            configs = {
                1: TampingConfig(
                    duration=15.0,
                    stop_time=4.0,
                    stop_duration=5.0,
                    speed=1.0,
                    terrain_type="industrial",
                ),
                2: TampingConfig(
                    duration=15.0,
                    stop_time=6.0,
                    stop_duration=4.0,
                    speed=0.9,
                    terrain_type="rural",
                ),
            }
            video_data = generate_tamping_video(configs[machine_id])
            VIDEO_CACHE[cache_key] = video_data
            results[cache_key] = "generated"
        else:
            results[cache_key] = "already cached"

    print("Cache warmup complete!")
    return jsonify({"status": "ok", "results": results})


@app.route("/api/cache/clear", methods=["POST"])
def clear_cache():
    """Clear all cached videos"""
    VIDEO_CACHE.clear()
    return jsonify({"status": "ok", "message": "Cache cleared"})


if __name__ == "__main__":
    print("=" * 60)
    print("Railroad Simulator API v2.0 - Enhanced Graphics")
    print("=" * 60)
    print("\nAvailable endpoints:")
    print("  GET  /api/health")
    print("  GET  /api/videos/info")
    print("  GET  /api/videos/track/1  - Gravel terrain")
    print("  GET  /api/videos/track/2  - Desert terrain")
    print("  GET  /api/videos/track/3  - Forest terrain")
    print("  GET  /api/videos/tamping/1 - Industrial tamping")
    print("  GET  /api/videos/tamping/2 - Rural tamping")
    print("  POST /api/cache/warmup    - Pre-generate all videos")
    print("  POST /api/cache/clear     - Clear video cache")
    print("=" * 60)
    print("\n💡 TIP: Run this to pre-generate videos:")
    print("   curl -X POST http://localhost:5000/api/cache/warmup")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=True)
