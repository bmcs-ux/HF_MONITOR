import requests
import json
import os
from datetime import datetime, timedelta

class NewsManager:
    def __init__(self, data_dir, logger):
        self.data_path = os.path.join(data_dir, "daily_news.json")
        self._log = logger
        self.high_impact_events = []
        self.last_sync_date = None

    def sync_news(self):
        """Mengambil berita dari API dan menyimpan ke JSON lokal."""
        try:
            self._log("[NEWS] Syncing daily economic calendar...")
            # Menggunakan API aggregator yang mengambil data ForexFactory
            # URL ini adalah contoh, pastikan menggunakan provider yang stabil
            url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
            response = requests.get(url, timeout=15)
            
            if response.status_code == 200:
                all_events = response.json()
                # Filter hanya High Impact dan sesuaikan waktu
                self.high_impact_events = [
                    e for e in all_events if e.get('impact') == 'High'
                ]
                
                with open(self.data_path, 'w') as f:
                    json.dump(self.high_impact_events, f)
                
                self.last_sync_date = datetime.now().date()
                self._log(f"[NEWS] Sync complete. Found {len(self.high_impact_events)} High Impact events.")
                return True
        except Exception as e:
            self._log(f"[ERROR] News sync failed: {e}")
            return False

    def load_local_news(self):
        """Memuat data berita dari file lokal ke memori."""
        if os.path.exists(self.data_path):
            with open(self.data_path, 'r') as f:
                self.high_impact_events = json.load(f)
            self._log(f"[NEWS] Loaded {len(self.high_impact_events)} events from local cache.")

    def is_currently_restricted(self):
        """Cek apakah sekarang masuk jendela dilarang (5 menit sebelum/sesudah)."""
        if not self.high_impact_events:
            return False

        # FundingPips mengacu pada waktu Eastern (ET) atau UTC tergantung server.
        # Mayoritas API menggunakan UTC. Pastikan jam VPS kamu sinkron.
        now_utc = datetime.utcnow()
        
        for event in self.high_impact_events:
            try:
                # Format FF API biasanya: "2026-01-18T10:00:00-05:00"
                event_time = self._parse_event_time(event['date'])
                
                start_window = event_time - timedelta(minutes=5)
                end_window = event_time + timedelta(minutes=5)

                if start_window <= now_utc <= end_window:
                    self._log(f"[BLOCK] Trade prohibited! News: {event['title']} ({event['country']})")
                    return True
            except:
                continue
        return False

    def _parse_event_time(self, date_str):
        # Helper untuk konversi berbagai format ISO ke UTC datetime
        # (Beberapa API menggunakan format yang sedikit berbeda)
        from dateutil import parser
        return parser.parse(date_str).astimezone(timezone.utc).replace(tzinfo=None)
