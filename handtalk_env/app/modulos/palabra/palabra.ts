import { Component, OnInit, OnDestroy, AfterViewInit } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-palabra',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './palabras.html',
  styleUrls: []
})
export class PalabraComponent implements OnInit, AfterViewInit, OnDestroy {
  readonly flaskUrl = 'http://localhost:5000';
  videoFeedUrl = `${this.flaskUrl}/video_feed`;
  
  targetWord: string = '-';
  detectedWord: string = '-';
  progress: string = '0/4';
  wordDetected: boolean = false;
  isCompleted: boolean = false;
  isRecording: boolean = false;
  
  private updateInterval: any;
  
  constructor(private http: HttpClient) {}
  
  ngOnInit() {
    console.log('PalabraComponent inicializado');
    // Establecer el modo en Flask cuando se carga el componente
    this.http.post<any>(`${this.flaskUrl}/set_mode`, { mode: 'words' }).subscribe({
      next: () => console.log('Modo establecido a words'),
      error: (error) => console.error('Error al establecer modo:', error)
    });
  }
  
  ngAfterViewInit() {
    console.log('PalabraComponent - DOM listo');
    this.updateInfo();
    this.updateInterval = setInterval(() => this.updateInfo(), 300);
  }
  
  ngOnDestroy() {
    if (this.updateInterval) {
      clearInterval(this.updateInterval);
    }
  }
  
  updateInfo() {
    this.http.get<any>(`${this.flaskUrl}/get_word_prediction`).subscribe({
      next: (data) => {
        console.log('Datos recibidos:', data);
        this.targetWord = data.target_word || '-';
        this.detectedWord = data.detected_word || '-';
        this.progress = data.progress || '0/4';
        this.wordDetected = data.word_detected || false;
        this.isRecording = data.recording || false;
      },
      error: (error) => {
        console.error('Error al obtener predicción:', error);
        this.detectedWord = '-';
        this.targetWord = '-';
      }
    });
  }
  
  nextWord() {
    this.http.post<any>(`${this.flaskUrl}/next_word`, {}).subscribe({
      next: (data) => {
        if (data.completed) {
          this.isCompleted = true;
          this.targetWord = '✓';
        } else {
          this.updateInfo();
        }
      },
      error: (error) => {
        console.error('Error al avanzar palabra:', error);
      }
    });
  }
  
  resetPractice() {
    this.http.post<any>(`${this.flaskUrl}/reset_word_practice`, {}).subscribe({
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
