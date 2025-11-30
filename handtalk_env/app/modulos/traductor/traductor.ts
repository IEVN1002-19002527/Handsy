import { Component, OnInit, OnDestroy, AfterViewInit, ViewChild, ElementRef } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';

@Component({
  selector: 'app-traductor',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './traductor.html',
  styleUrls: []
})
export class TraductorComponent implements OnInit, AfterViewInit, OnDestroy {
  readonly flaskUrl = 'http://localhost:5000';
  videoFeedUrl = `${this.flaskUrl}/video_feed`;
  
  currentDetection: string = '-';
  translationText: string = '';
  isDetecting: boolean = false;
  
  private updateInterval: any;
  private lastDetection: string = '';
  private detectionCount: number = 0;
  private readonly DETECTION_THRESHOLD = 3;
  
  @ViewChild('translationTextarea', { static: false }) translationTextarea!: ElementRef<HTMLTextAreaElement>;
  
  constructor(private http: HttpClient) {}
  
  ngOnInit() {
    console.log('TraductorComponent inicializado');
    // Establecer el modo en Flask cuando se carga el componente
    this.http.post<any>(`${this.flaskUrl}/set_mode`, { mode: 'translator' }).subscribe({
      next: () => console.log('Modo establecido a translator'),
      error: (error) => console.error('Error al establecer modo:', error)
    });
  }
  
  ngAfterViewInit() {
    console.log('TraductorComponent - DOM listo');
    this.updateInfo();
    this.updateInterval = setInterval(() => this.updateInfo(), 200);
  }
  
  ngOnDestroy() {
    if (this.updateInterval) {
      clearInterval(this.updateInterval);
    }
  }
  
  updateInfo() {
    this.http.get<any>(`${this.flaskUrl}/get_translator_prediction`).subscribe({
      next: (data) => {
        const currentDetection = data.prediction || '-';
        
        if (currentDetection !== '-' && 
            currentDetection !== 'Esperando mano...' && 
            currentDetection !== 'Mano detectada, procesando...' && 
            currentDetection !== 'Grabando video...') {
          this.currentDetection = currentDetection;
          this.isDetecting = true;
          
          if (currentDetection === this.lastDetection) {
            this.detectionCount++;
            if (this.detectionCount >= this.DETECTION_THRESHOLD && this.lastDetection !== '') {
              let valueToAdd = currentDetection;
              if (currentDetection.includes(':')) {
                valueToAdd = currentDetection.split(':')[1].trim();
              }
              
              const lastChar = this.translationText.slice(-1);
              if (valueToAdd !== lastChar && valueToAdd !== '-') {
                this.translationText += valueToAdd;
                this.detectionCount = 0;
              }
            }
          } else {
            this.detectionCount = 0;
          }
          this.lastDetection = currentDetection;
        } else {
          this.currentDetection = '-';
          this.isDetecting = false;
          this.detectionCount = 0;
          this.lastDetection = '';
        }
      },
      error: (error) => {
        console.error('Error al obtener predicción:', error);
        this.currentDetection = '-';
        this.isDetecting = false;
      }
    });
  }
  
  addSpace() {
    this.translationText += ' ';
  }
  
  clearText() {
    if (confirm('¿Estás seguro de que deseas limpiar todo el texto?')) {
      this.translationText = '';
      this.lastDetection = '';
      this.detectionCount = 0;
    }
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
