import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
from enum import Enum


class UserArchetype(Enum):
    """Define distinct user behavior archetypes"""

    METICULOUS_PLANNER = "meticulous_planner"
    HURRIED_MULTITASKER = "hurried_multitasker"
    NOVICE_USER = "novice_user"
    TECHNICAL_EXPERT = "technical_expert"
    CASUAL_USER = "casual_user"


class ActivityFlow(Enum):
    """Define realistic activity flow patterns"""

    NORMAL_ROUTINE = "normal_routine"
    DATA_ACCESS_HEAVY = "data_access_heavy"
    EMAIL_FOCUSED = "email_focused"
    SYSTEM_ADMIN = "system_admin"
    SUSPICIOUS_EXFILTRATION = "suspicious_exfiltration"


class BiometricSyntheticGenerator:

    def __init__(self, seed_data_path=None):
        np.random.seed(42)
        self.seed_data = (
            self._load_seed_data(seed_data_path) if seed_data_path else None
        )

        self.user_agents = {
            "windows_chrome": [
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
            ],
            "windows_firefox": [
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0",
            ],
            "mac_chrome": [
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            ],
            "mac_safari": [
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15",
            ],
            "linux": [
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            ],
            "mobile_ios": [
                "Mozilla/5.0 (iPhone; CPU iPhone OS 17_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1",
                "Mozilla/5.0 (iPad; CPU OS 17_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1",
            ],
        }

        # IP address pools
        self.ip_ranges = {
            "internal": ["192.168.1", "192.168.2", "10.0.1", "10.0.2", "10.0.3"],
            "vpn": ["172.16.10", "172.16.11", "172.16.12"],
            "external": ["203.0.113", "198.51.100", "45.33.32"],
        }

        # Archetype behavioral parameters
        self.archetype_params = {
            UserArchetype.METICULOUS_PLANNER: {
                "wpm_range": (35, 55),
                "accuracy_range": (95, 99),
                "mouse_velocity_range": (0.08, 0.15),
                "thinking_pause_freq": 0.3,
                "login_time_consistency": 0.95,
                "typical_hours": [8, 13, 16],
                "session_duration_range": (180000, 300000),
            },
            UserArchetype.HURRIED_MULTITASKER: {
                "wpm_range": (65, 90),
                "accuracy_range": (75, 85),
                "mouse_velocity_range": (0.20, 0.35),
                "thinking_pause_freq": 0.05,
                "login_time_consistency": 0.70,
                "typical_hours": [7, 9, 12, 15, 18],
                "session_duration_range": (60000, 150000),
            },
            UserArchetype.NOVICE_USER: {
                "wpm_range": (20, 35),
                "accuracy_range": (70, 80),
                "mouse_velocity_range": (0.05, 0.12),
                "thinking_pause_freq": 0.45,
                "login_time_consistency": 0.85,
                "typical_hours": [9, 14],
                "session_duration_range": (200000, 400000),
            },
            UserArchetype.TECHNICAL_EXPERT: {
                "wpm_range": (70, 100),
                "accuracy_range": (90, 97),
                "mouse_velocity_range": (0.15, 0.25),
                "thinking_pause_freq": 0.10,
                "login_time_consistency": 0.75,
                "typical_hours": [10, 14, 16, 20],
                "session_duration_range": (150000, 250000),
            },
            UserArchetype.CASUAL_USER: {
                "wpm_range": (40, 60),
                "accuracy_range": (80, 90),
                "mouse_velocity_range": (0.10, 0.20),
                "thinking_pause_freq": 0.20,
                "login_time_consistency": 0.80,
                "typical_hours": [9, 12, 15],
                "session_duration_range": (120000, 240000),
            },
        }

    def _load_seed_data(self, path):
        with open(path, "r") as f:
            return json.load(f)

    def _select_archetype(self, user_id, is_insider=False):
        """Select user archetype with weighted probabilities"""
        archetypes = list(UserArchetype)

        if is_insider:
            # Insiders more likely to be technical or multitaskers
            weights = [0.1, 0.3, 0.05, 0.4, 0.15]
        else:
            # Normal distribution
            weights = [0.2, 0.25, 0.15, 0.2, 0.2]

        np.random.seed(hash(f"{user_id}_archetype") % (2**32))
        archetype = np.random.choice(archetypes, p=weights)
        np.random.seed()

        return archetype

    def _determine_activity_flow(
        self, session_idx, total_sessions, is_insider, archetype
    ):
        """Determine activity flow for a session based on context"""

        if is_insider:
            # Progressive insider threat: normal -> suspicious
            progression = session_idx / total_sessions

            if progression < 0.3:
                # Early phase: establish normal baseline
                return ActivityFlow.NORMAL_ROUTINE
            elif progression < 0.6:
                # Middle phase: gradual increase in data access
                return np.random.choice(
                    [ActivityFlow.NORMAL_ROUTINE, ActivityFlow.DATA_ACCESS_HEAVY],
                    p=[0.6, 0.4],
                )
            else:
                # Late phase: suspicious behavior increases
                return np.random.choice(
                    [
                        ActivityFlow.NORMAL_ROUTINE,
                        ActivityFlow.DATA_ACCESS_HEAVY,
                        ActivityFlow.SUSPICIOUS_EXFILTRATION,
                    ],
                    p=[0.3, 0.4, 0.3],
                )
        else:
            # Normal users: mostly routine with occasional variations
            if archetype == UserArchetype.TECHNICAL_EXPERT:
                return np.random.choice(
                    [
                        ActivityFlow.NORMAL_ROUTINE,
                        ActivityFlow.SYSTEM_ADMIN,
                        ActivityFlow.DATA_ACCESS_HEAVY,
                    ],
                    p=[0.5, 0.3, 0.2],
                )
            else:
                return np.random.choice(
                    [
                        ActivityFlow.NORMAL_ROUTINE,
                        ActivityFlow.EMAIL_FOCUSED,
                        ActivityFlow.DATA_ACCESS_HEAVY,
                    ],
                    p=[0.6, 0.3, 0.1],
                )

    def _get_flow_characteristics(self, flow):
        """Get behavioral characteristics for each activity flow"""
        flow_params = {
            ActivityFlow.NORMAL_ROUTINE: {
                "off_hours_prob": 0.05,
                "external_ip_prob": 0.02,
                "device_change_prob": 0.01,
                "failed_login_prob": 0.02,
                "email_recipients_external": 0.1,
                "data_access_volume": "low",
                "stress_multiplier": 1.0,
            },
            ActivityFlow.DATA_ACCESS_HEAVY: {
                "off_hours_prob": 0.15,
                "external_ip_prob": 0.08,
                "device_change_prob": 0.05,
                "failed_login_prob": 0.05,
                "email_recipients_external": 0.25,
                "data_access_volume": "high",
                "stress_multiplier": 1.15,
            },
            ActivityFlow.EMAIL_FOCUSED: {
                "off_hours_prob": 0.08,
                "external_ip_prob": 0.03,
                "device_change_prob": 0.02,
                "failed_login_prob": 0.03,
                "email_recipients_external": 0.4,
                "data_access_volume": "low",
                "stress_multiplier": 0.95,
            },
            ActivityFlow.SYSTEM_ADMIN: {
                "off_hours_prob": 0.25,
                "external_ip_prob": 0.15,
                "device_change_prob": 0.10,
                "failed_login_prob": 0.08,
                "email_recipients_external": 0.15,
                "data_access_volume": "medium",
                "stress_multiplier": 1.1,
            },
            ActivityFlow.SUSPICIOUS_EXFILTRATION: {
                "off_hours_prob": 0.70,
                "external_ip_prob": 0.60,
                "device_change_prob": 0.40,
                "failed_login_prob": 0.20,
                "email_recipients_external": 0.80,
                "data_access_volume": "very_high",
                "stress_multiplier": 1.4,  # More stress = faster typing, erratic mouse
            },
        }
        return flow_params[flow]

    def generate_device_fingerprint(self, user_profile, flow_params):
        """Generate device fingerprint based on flow characteristics"""
        base_fingerprint = user_profile["baseline"]["device_fingerprint"]

        if np.random.random() < flow_params["device_change_prob"]:
            return f"FP_{np.random.randint(1000, 9999)}"
        return base_fingerprint

    def generate_user_agent(self, user_profile, flow_params):
        """Generate user agent with device consistency"""
        device_category = user_profile["baseline"]["device_category"]

        if np.random.random() < flow_params["device_change_prob"]:
            new_category = np.random.choice(list(self.user_agents.keys()))
            return np.random.choice(self.user_agents[new_category])

        base_ua = user_profile["baseline"]["user_agent"]

        if np.random.random() < 0.03:  # 3% chance of version update
            import re

            version_match = re.search(r"(\d+)\.0\.0\.0", base_ua)
            if version_match:
                old_version = version_match.group(1)
                new_version = str(int(old_version) + np.random.randint(1, 3))
                return base_ua.replace(f"{old_version}.0.0.0", f"{new_version}.0.0.0")

        return base_ua

    def generate_ip_address(self, user_profile, flow_params):
        """Generate IP with location consistency"""
        typical_range = user_profile["baseline"]["ip_range"]

        if np.random.random() < flow_params["external_ip_prob"]:
            # Suspicious: external IP
            range_choice = np.random.choice(self.ip_ranges["external"])
        elif np.random.random() < 0.15:
            # VPN usage (acceptable for remote work)
            range_choice = np.random.choice(self.ip_ranges["vpn"])
        else:
            # Normal: consistent internal network
            range_choice = typical_range

        return f"{range_choice}.{np.random.randint(1, 255)}"

    def generate_typing_dynamics(self, user_profile, archetype_params, flow_params):
        """Generate typing patterns based on archetype and stress level"""
        wpm_min, wpm_max = archetype_params["wpm_range"]
        base_wpm = np.random.uniform(wpm_min, wpm_max)

        # Adjust for stress/hurry in suspicious activities
        stress_multiplier = flow_params["stress_multiplier"]
        if stress_multiplier > 1.2:
            base_wpm *= np.random.uniform(1.1, 1.3)

        n_keystrokes = int(np.random.uniform(30, 80))

        # Dwell times based on archetype
        dwell_mean = user_profile["baseline"]["dwell_mean"]
        dwell_std = user_profile["baseline"]["dwell_std"]

        # Increase variance under stress
        dwell_std_adjusted = dwell_std * stress_multiplier

        dwell_times = np.random.gamma(
            shape=max(1, (dwell_mean / dwell_std_adjusted) ** 2),
            scale=dwell_std_adjusted**2 / max(1, dwell_mean),
            size=n_keystrokes,
        ).astype(int)
        dwell_times = np.clip(dwell_times, 50, 500)

        # Flight times with realistic pauses
        flight_times = []
        thinking_pause_freq = archetype_params["thinking_pause_freq"]

        # Under stress, fewer thinking pauses
        thinking_pause_freq = thinking_pause_freq / stress_multiplier

        for i in range(n_keystrokes):
            if np.random.random() < thinking_pause_freq:
                # Thinking pause
                flight = np.random.uniform(1000, 5000)
            elif np.random.random() < 0.9:
                # Normal inter-keystroke
                flight = np.random.exponential(scale=200)
            else:
                # Long pause (distraction, phone call, etc.)
                flight = np.random.exponential(scale=3000)
            flight_times.append(int(flight))

        # Accuracy decreases under stress
        acc_min, acc_max = archetype_params["accuracy_range"]
        accuracy = np.random.uniform(acc_min, acc_max) / stress_multiplier
        accuracy = np.clip(accuracy, 60, 100)

        return {
            "wpm": int(base_wpm),
            "dwellTime": dwell_times.tolist(),
            "flightTime": flight_times,
            "accuracy": round(accuracy, 1),
        }

    def generate_mouse_dynamics(self, user_profile, archetype_params, flow_params):
        """Generate mouse patterns based on archetype and stress"""
        vel_min, vel_max = archetype_params["mouse_velocity_range"]
        base_velocity = np.random.uniform(vel_min, vel_max)

        # Increase velocity under stress
        stress_multiplier = flow_params["stress_multiplier"]
        base_velocity *= stress_multiplier

        # Acceleration correlates with velocity
        base_accel = base_velocity * np.random.uniform(1.5, 2.5)

        # More mouse points for slower, deliberate users
        n_points = int(np.random.uniform(15, 40))

        # Generate Bezier curve path
        start_x, start_y = np.random.randint(100, 900), np.random.randint(100, 600)
        end_x, end_y = np.random.randint(100, 900), np.random.randint(100, 600)

        ctrl1_x = start_x + np.random.normal(0, 150)
        ctrl1_y = start_y + np.random.normal(0, 150)
        ctrl2_x = end_x + np.random.normal(0, 150)
        ctrl2_y = end_y + np.random.normal(0, 150)

        click_pattern = []
        base_timestamp = int(datetime.now().timestamp() * 1000)

        for i in range(n_points):
            t = i / (n_points - 1)

            x = (
                (1 - t) ** 3 * start_x
                + 3 * (1 - t) ** 2 * t * ctrl1_x
                + 3 * (1 - t) * t**2 * ctrl2_x
                + t**3 * end_x
            )
            y = (
                (1 - t) ** 3 * start_y
                + 3 * (1 - t) ** 2 * t * ctrl1_y
                + 3 * (1 - t) * t**2 * ctrl2_y
                + t**3 * end_y
            )

            # Add jitter (hand tremor simulation) - increases under stress
            jitter_base = (
                2.0 if user_profile["archetype"] == UserArchetype.NOVICE_USER else 1.0
            )
            jitter = jitter_base * stress_multiplier
            x += np.random.normal(0, jitter)
            y += np.random.normal(0, jitter)

            time_delta = int(np.random.gamma(2, 10))

            click_pattern.append(
                {"x": int(x), "y": int(y), "timestamp": base_timestamp + i * time_delta}
            )

        curvature = user_profile["baseline"]["curvature"]
        # More erratic movements under stress
        curvature *= stress_multiplier
        curvature = min(curvature, 1.0)

        idle_time = int(np.random.exponential(1500))

        return {
            "velocity": round(base_velocity, 4),
            "acceleration": round(base_accel, 2),
            "clickPattern": click_pattern,
            "movementCurvature": round(curvature, 2),
            "idleTime": idle_time,
        }

    def generate_email_context(self, user_profile, flow_params):
        """Generate email patterns based on flow"""
        typical_hours = user_profile["baseline"]["typical_hours"]

        if flow_params["off_hours_prob"] > 0.5:
            # Suspicious: off-hours activity
            send_times = list(np.random.choice([0, 1, 2, 3, 4, 5, 22, 23], size=3))
        else:
            # Normal: typical work hours with variation
            send_times = [int(h + np.random.normal(0, 1.5)) % 24 for h in typical_hours]

        # Recipient patterns based on external communication probability
        num_recipients = int(np.random.poisson(3) + 1)
        external_ratio = flow_params["email_recipients_external"]

        recipients = []
        for _ in range(num_recipients):
            if np.random.random() < external_ratio:
                recipients.append(
                    f"external_{np.random.randint(1, 100)}@competitor.com"
                )
            else:
                recipients.append(f"colleague_{np.random.randint(1, 200)}@company.com")

        if flow_params["data_access_volume"] in ["high", "very_high"]:
            email_length = int(np.random.uniform(800, 2500))
            subject_complexity = int(np.random.uniform(60, 90))
        else:
            email_length = int(np.random.normal(300, 150))
            subject_complexity = int(np.random.normal(45, 20))

        return {
            "typicalSendTimes": send_times,
            "recipientPatterns": recipients,
            "subjectComplexity": np.clip(subject_complexity, 0, 100),
            "emailLength": max(50, email_length),
            "attachmentCount": int(np.random.poisson(1))
            if flow_params["data_access_volume"] == "very_high"
            else 0,
        }

    def generate_touch_gesture(self, flow_params):
        """Generate touch gesture data for mobile devices"""
        stress_multiplier = flow_params["stress_multiplier"]

        pressure = int(np.random.uniform(40, 80) * stress_multiplier)
        swipe_velocity = int(np.random.uniform(30, 70) * stress_multiplier)
        tap_duration = int(np.random.uniform(80, 200) / stress_multiplier)
        finger_area = int(np.random.uniform(40, 70))

        return {
            "pressure": min(pressure, 100),
            "swipeVelocity": min(swipe_velocity, 100),
            "tapDuration": tap_duration,
            "fingerArea": finger_area,
        }

    def generate_session_pattern(
        self, user_profile, archetype_params, flow_params, session_date
    ):
        """Generate session timing patterns"""
        typical_hour = user_profile["baseline"]["typical_login_hour"]

        if np.random.random() < flow_params["off_hours_prob"]:
            # Off-hours login
            hour = int(np.random.choice([0, 1, 2, 3, 4, 5, 22, 23]))
            day = int(np.random.choice([6, 0]))  # Weekend
        else:
            # Normal hours
            hour = int(typical_hour + np.random.normal(0, 1)) % 24
            day = session_date.weekday()

        # Session duration based on archetype
        dur_min, dur_max = archetype_params["session_duration_range"]
        duration = int(np.random.uniform(dur_min, dur_max))

        # Failed attempts
        if np.random.random() < flow_params["failed_login_prob"]:
            failed_attempts = int(np.random.poisson(1.5) + 1)
        else:
            failed_attempts = 0

        base_consistency = archetype_params["login_time_consistency"] * 100
        location_consistency = int(base_consistency + np.random.normal(0, 5))

        return {
            "timeOfDay": hour,
            "dayOfWeek": day,
            "loginDuration": duration,
            "failedAttempts": failed_attempts,
            "locationConsistency": np.clip(location_consistency, 0, 100),
            "sessionDate": session_date.isoformat(),
        }

    def calculate_click_frequency(self, click_pattern):
        """Calculate clicks per second from click pattern"""
        if len(click_pattern) < 2:
            return 0.0

        time_span = (
            click_pattern[-1]["timestamp"] - click_pattern[0]["timestamp"]
        ) / 1000.0
        if time_span == 0:
            return 0.0

        return round(len(click_pattern) / time_span, 2)

    def calculate_scroll_speed(self, mouse_dynamics):
        """Derive scroll speed from mouse velocity"""
        base_speed = mouse_dynamics["velocity"] * 100
        return round(base_speed + np.random.normal(0, 5), 2)

    def generate_user_profile(self, user_id, is_insider=False):
        """Generate complete user profile with archetype"""
        np.random.seed(hash(f"{user_id}_profile") % (2**32))

        archetype = self._select_archetype(user_id, is_insider)
        archetype_params = self.archetype_params[archetype]

        device_categories = list(self.user_agents.keys())
        device_weights = [0.35, 0.10, 0.25, 0.10, 0.08, 0.12]
        device_category = np.random.choice(device_categories, p=device_weights)

        wpm_min, wpm_max = archetype_params["wpm_range"]

        profile = {
            "userId": user_id,
            "is_insider": is_insider,
            "archetype": archetype,
            "baseline": {
                "wpm": np.random.uniform(wpm_min, wpm_max),
                "dwell_mean": np.random.uniform(100, 200),
                "dwell_std": np.random.uniform(30, 70),
                "velocity": np.random.uniform(
                    *archetype_params["mouse_velocity_range"]
                ),
                "acceleration": np.random.uniform(0.3, 0.7),
                "curvature": np.random.beta(2, 5),
                "typical_hours": archetype_params["typical_hours"],
                "typical_login_hour": int(
                    np.random.choice(archetype_params["typical_hours"])
                ),
                "device_fingerprint": f"FP_{np.random.randint(1000, 9999)}",
                "device_category": device_category,
                "user_agent": np.random.choice(self.user_agents[device_category]),
                "ip_range": np.random.choice(self.ip_ranges["internal"]),
            },
        }

        np.random.seed()
        return profile

    def generate_session(self, user_profile, session_idx, total_sessions, session_date):
        """Generate complete session with activity flow - NO anomalyScore or riskLevel"""
        archetype = user_profile["archetype"]
        archetype_params = self.archetype_params[archetype]

        activity_flow = self._determine_activity_flow(
            session_idx, total_sessions, user_profile["is_insider"], archetype
        )
        flow_params = self._get_flow_characteristics(activity_flow)

        is_threat = activity_flow == ActivityFlow.SUSPICIOUS_EXFILTRATION or (
            activity_flow == ActivityFlow.DATA_ACCESS_HEAVY and np.random.random() < 0.3
        )

        mouse_dynamics = self.generate_mouse_dynamics(
            user_profile, archetype_params, flow_params
        )

        typing_speed = self.generate_typing_dynamics(
            user_profile, archetype_params, flow_params
        )

        session = {
            "userId": user_profile["userId"],
            "email": f"user{user_profile['userId']}@unifinance.com",
            "sessionId": f"session_{np.random.randint(100000, 999999)}",
            "archetype": archetype.value,
            "activityFlow": activity_flow.value,
            "logonPattern": self.generate_session_pattern(
                user_profile, archetype_params, flow_params, session_date
            ),
            "typingSpeed": typing_speed,
            "mouseDynamics": mouse_dynamics,
            "emailContext": self.generate_email_context(user_profile, flow_params),
            "touchGesture": self.generate_touch_gesture(flow_params),
            "deviceFingerprint": self.generate_device_fingerprint(
                user_profile, flow_params
            ),
            "ipAddress": self.generate_ip_address(user_profile, flow_params),
            "userAgent": self.generate_user_agent(user_profile, flow_params),
            "clickFrequency": self.calculate_click_frequency(
                mouse_dynamics["clickPattern"]
            ),
            "scrollSpeed": self.calculate_scroll_speed(mouse_dynamics),
            "label": 1 if is_threat else 0,
            "createdAt": session_date.isoformat(),
        }

        return session

    def generate_dataset(self, n_users=5000, n_insiders=500, sessions_per_user=100):
        """Generate complete dataset"""
        print("\n" + "=" * 80)
        print("GENERATING SYNTHETIC BIOMETRIC DATASET")
        print("=" * 80)
        print(f"Total Users: {n_users:,}")
        print(f"Insider Threats: {n_insiders:,} ({n_insiders / n_users * 100:.1f}%)")
        print(f"Sessions per User: {sessions_per_user}")
        print(f"Total Sessions: {n_users * sessions_per_user:,}")
        print("=" * 80 + "\n")

        dataset = []
        insider_ids = np.random.choice(range(n_users), size=n_insiders, replace=False)

        base_date = datetime.now() - timedelta(days=180)

        for user_id in range(n_users):
            if user_id % 500 == 0:
                print(f"  Processing user {user_id}/{n_users}...")

            is_insider = user_id in insider_ids
            user_profile = self.generate_user_profile(user_id, is_insider)

            for session_idx in range(sessions_per_user):
                days_offset = (session_idx / sessions_per_user) * 180
                session_date = base_date + timedelta(days=days_offset)

                session = self.generate_session(
                    user_profile, session_idx, sessions_per_user, session_date
                )
                dataset.append(session)

        print("\n✅ Dataset generation complete!")
        print(f"   Total records: {len(dataset):,}")

        # label distribution
        threats = sum(1 for s in dataset if s["label"] == 1)
        normal = len(dataset) - threats
        print(f"   Normal sessions: {normal:,} ({normal / len(dataset) * 100:.1f}%)")
        print(f"   Threat sessions: {threats:,} ({threats / len(dataset) * 100:.1f}%)")

        return dataset

    def _flatten_session(self, session):
        """Flatten nested session data for CSV - EXCLUDING anomalyScore and riskLevel"""
        flat = {
            "userId": session["userId"],
            "email": session["email"],
            "sessionId": session["sessionId"],
            "archetype": session["archetype"],
            "activityFlow": session["activityFlow"],
            "createdAt": session["createdAt"],
            "label": session["label"],
            # Device & Network
            "deviceFingerprint": session["deviceFingerprint"],
            "ipAddress": session["ipAddress"],
            "userAgent": session["userAgent"],
            # Logon Pattern
            "timeOfDay": session["logonPattern"]["timeOfDay"],
            "dayOfWeek": session["logonPattern"]["dayOfWeek"],
            "loginDuration": session["logonPattern"]["loginDuration"],
            "failedAttempts": session["logonPattern"]["failedAttempts"],
            "locationConsistency": session["logonPattern"]["locationConsistency"],
            "sessionDate": session["logonPattern"]["sessionDate"],
            # Typing Speed
            "wpm": session["typingSpeed"]["wpm"],
            "typingAccuracy": session["typingSpeed"]["accuracy"],
            "dwellTime_mean": np.mean(session["typingSpeed"]["dwellTime"]),
            "dwellTime_std": np.std(session["typingSpeed"]["dwellTime"]),
            "flightTime_mean": np.mean(session["typingSpeed"]["flightTime"]),
            "flightTime_std": np.std(session["typingSpeed"]["flightTime"]),
            "maxFlightTime": np.max(session["typingSpeed"]["flightTime"]),
            # Mouse Dynamics
            "mouseDynamics_velocity": session["mouseDynamics"]["velocity"],
            "mouseDynamics_acceleration": session["mouseDynamics"]["acceleration"],
            "mouseDynamics_curvature": session["mouseDynamics"]["movementCurvature"],
            "idleTime": session["mouseDynamics"]["idleTime"],
            "clickFrequency": session["clickFrequency"],
            "scrollSpeed": session["scrollSpeed"],
            # Touch Gesture
            "touchPressure": session["touchGesture"]["pressure"],
            "swipeVelocity": session["touchGesture"]["swipeVelocity"],
            "tapDuration": session["touchGesture"]["tapDuration"],
            "fingerArea": session["touchGesture"]["fingerArea"],
            # Email Context
            "emailLength": session["emailContext"]["emailLength"],
            "subjectComplexity": session["emailContext"]["subjectComplexity"],
            "numRecipients": len(session["emailContext"]["recipientPatterns"]),
            "attachmentCount": session["emailContext"].get("attachmentCount", 0),
        }

        return flat

    def save_dataset(self, dataset, output_path):
        """Save dataset to JSON with proper type conversion"""
        def convert(o):
            if isinstance(o, (np.integer,)):
                return int(o)
            elif isinstance(o, (np.floating,)):
                return float(o)
            elif isinstance(o, np.ndarray):
                return o.tolist()
            elif isinstance(o, (datetime, np.datetime64)):
                return str(o)
            elif isinstance(o, Enum):
                return o.value
            return o

        print(f"\n💾 Saving JSON to {output_path}...")
        import os

        with open(output_path, "w") as f:
            json.dump(dataset, f, indent=2, default=convert)

        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"✅ JSON saved: {len(dataset):,} records ({file_size_mb:.2f} MB)")

    def save_dataset_csv(self, dataset, output_path):
        """Save dataset to CSV with flattened structure"""
        print(f"\n💾 Saving CSV to {output_path}...")

        flattened_data = []
        for i, session in enumerate(dataset):
            if i % 10000 == 0 and i > 0:
                print(f"   Flattening progress: {i:,}/{len(dataset):,}")
            flattened_data.append(self._flatten_session(session))

        df = pd.DataFrame(flattened_data)
        df.to_csv(output_path, index=False)

        import os
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(
            f"✅ CSV saved: {df.shape[0]:,} rows × {df.shape[1]} columns ({file_size_mb:.2f} MB)"
        )

        # Show feature summary
        exclude_cols = {
            "userId",
            "email",
            "sessionId",
            "archetype",
            "activityFlow",
            "createdAt",
            "sessionDate",
            "label",
            "deviceFingerprint",
            "ipAddress",
            "userAgent",
        }
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        print(f"   Numerical features: {len(feature_cols)}")
        print(f"   Sample features: {feature_cols[:10]}")

        # Label distribution
        print("\n📈 Label Distribution:")
        print(df["label"].value_counts().to_string())
        print(f"   Threat ratio: {df['label'].mean() * 100:.2f}%")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("=" * 80)

    generator = BiometricSyntheticGenerator()

    # training dataset
    print("\n=== Generating Training Dataset ===")
    train_dataset = generator.generate_dataset(
        n_users=2000,  # 2,000 users
        n_insiders=200,  # 10% insider threats (200 users)
        sessions_per_user=100,  # 100 sessions per user = 200,000 total
    )

    # test dataset
    print("\n=== Generating Test Dataset ===")
    test_dataset = generator.generate_dataset(
        n_users=1000,  # 1,000 users
        n_insiders=100,  # 10% insider threats
        sessions_per_user=100,  # 100 sessions per user = 100,000 total
    )

    # Save datasets
    generator.save_dataset_csv(train_dataset, "biometric_train_v2.csv")
    generator.save_dataset_csv(test_dataset, "biometric_test_v2.csv")

    # generator.save_dataset(train_dataset, "biometric_train.json")
    # generator.save_dataset(test_dataset, "biometric_test.json")