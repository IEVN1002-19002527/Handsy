import { Component, OnInit, OnDestroy, AfterViewInit } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-mes',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './meses.html',
  styleUrls: []
})
export class MesComponent implements OnInit, AfterViewInit, OnDestroy {
  readonly flaskUrl = 'http://localhost:5000';
  videoFeedUrl = `${this.flaskUrl}/video_feed`;
  
  targetMonth: string = '-';
  detectedMonth: string = '-';
  progress: string = '0/4';
  monthDetected: boolean = false;
  isCompleted: boolean = false;
  isRecording: boolean = false;
  
  private updateInterval: any;
  
  constructor(private http: HttpClient) {}
  
  ngOnInit() {
    console.log('MesComponent inicializado');
    // Establecer el modo en Flask cuando se carga el componente
    this.http.post<any>(`${this.flaskUrl}/set_mode`, { mode: 'months' }).subscribe({
      next: () => console.log('Modo establecido a months'),
      error: (error) => console.error('Error al establecer modo:', error)
    });
  }
  
  ngAfterViewInit() {
    console.log('MesComponent - DOM listo');
    this.updateInfo();
    this.updateInterval = setInterval(() => this.updateInfo(), 300);
  }
  
  ngOnDestroy() {
    if (this.updateInterval) {
      clearInterval(this.updateInterval);
    }
  }
  
  updateInfo() {
    this.http.get<any>(`${this.flaskUrl}/get_month_prediction`).subscribe({
      next: (data) => {
        console.log('Datos recibidos:', data);
        this.targetMonth = data.target_month || '-';
        this.detectedMonth = data.detected_month || '-';
        this.progress = data.progress || '0/4';
        this.monthDetected = data.month_detected || false;
        this.isRecording = data.recording || false;
      },
      error: (error) => {
        console.error('Error al obtener predicción:', error);
        this.detectedMonth = '-';
        this.targetMonth = '-';
      }
    });
  }
  
  nextMonth() {
    this.http.post<any>(`${this.flaskUrl}/next_month`, {}).subscribe({
      next: (data) => {
        if (data.completed) {
          this.isCompleted = true;
          this.targetMonth = '✓';
        } else {
          this.updateInfo();
        }
      },
      error: (error) => {
        console.error('Error al avanzar mes:', error);
      }
    });
  }
  
  resetPractice() {
    this.http.post<any>(`${this.flaskUrl}/reset_month_practice`, {}).subscribe({
      next: (data) => {
        this.isCompleted = false;
        this.updateInfo();
      },
      error: (error) => {
        console.error('Error al reiniciar práctica:', error);
      }
    });
  }
  
  onImageError(event: Event): void {
    const img = event.target as HTMLImageElement;
    if (img) {
      img.src = 'data:image/svg+xml,%3Csvg xmlns=\'http://www.w3.org/2000/svg\' width=\'640\' height=\'480\'%3E%3Crect fill=\'%23000\' width=\'640\' height=\'480\'/%3E%3Ctext x=\'50%25\' y=\'50%25\' fill=\'white\' font-size=\'20\' text-anchor=\'middle\' dominant-baseline=\'middle\'%3EError cargando video%3C/text%3E%3C/svg%3E';
    }
  }
  
  url_for(endpoint: string): string {
    const routes: { [key: string]: string } = {
      'video_feed': `${this.flaskUrl}/video_feed`
    };
    return routes[endpoint] || `${this.flaskUrl}/${endpoint}`;
  }
}
