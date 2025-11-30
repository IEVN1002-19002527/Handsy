import { Component, OnInit, OnDestroy, AfterViewInit } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-letra',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './letras.html',
  styleUrls: []
})
export class LetraComponent implements OnInit, AfterViewInit, OnDestroy {
  // URL del servidor Flask
  readonly flaskUrl = 'http://localhost:5000';
  videoFeedUrl = `${this.flaskUrl}/video_feed`;
  
  targetLetter: string = '-';
  detectedLetter: string = '-';
  progress: string = '0/14';
  letterDetected: boolean = false;
  isCompleted: boolean = false;
  
  private updateInterval: any;
  
  constructor(private http: HttpClient) {}
  
  ngOnInit() {
    console.log('LetraComponent inicializado');
    // Establecer el modo en Flask cuando se carga el componente
    this.http.post<any>(`${this.flaskUrl}/set_mode`, { mode: 'letters' }).subscribe({
      next: () => console.log('Modo establecido a letters'),
      error: (error) => console.error('Error al establecer modo:', error)
    });
  }
  
  ngAfterViewInit() {
    console.log('LetraComponent - DOM listo');
    this.updateInfo();
    // Actualizar cada 300 ms
    this.updateInterval = setInterval(() => this.updateInfo(), 300);
  }
  
  ngOnDestroy() {
    if (this.updateInterval) {
      clearInterval(this.updateInterval);
    }
  }
  
  updateInfo() {
    this.http.get<any>(`${this.flaskUrl}/get_prediction`).subscribe({
      next: (data) => {
        console.log('Datos recibidos:', data);
        this.targetLetter = data.target_letter || '-';
        this.detectedLetter = data.detected_letter || '-';
        this.progress = data.progress || '0/14';
        this.letterDetected = data.letter_detected || false;
      },
      error: (error) => {
        console.error('Error al obtener predicción:', error);
        this.detectedLetter = '-';
        this.targetLetter = '-';
      }
    });
  }
  
  nextLetter() {
    this.http.post<any>(`${this.flaskUrl}/next_letter`, {}).subscribe({
      next: (data) => {
        if (data.completed) {
          this.isCompleted = true;
          this.targetLetter = '✓';
        } else {
          this.updateInfo();
        }
      },
      error: (error) => {
        console.error('Error al avanzar letra:', error);
      }
    });
  }
  
  resetPractice() {
    this.http.post<any>(`${this.flaskUrl}/reset_practice`, {}).subscribe({
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
    // Mapeo de endpoints de Flask a URLs
    const routes: { [key: string]: string } = {
      'video_feed': `${this.flaskUrl}/video_feed`
    };
    return routes[endpoint] || `${this.flaskUrl}/${endpoint}`;
  }
}
