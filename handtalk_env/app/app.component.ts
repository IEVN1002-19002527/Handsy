import { RouterOutlet } from '@angular/router';
import { Navbar } from './navbar/navbar';
import { Component, OnInit } from '@angular/core';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [RouterOutlet, Navbar],
  templateUrl: './app.html',
  styleUrl: './app.css'
})
export class AppComponent implements OnInit {
  
    title= 'web-app';

    ngOnInit(): void {
      // Inicializar Flowbite si está disponible
      if (typeof window !== 'undefined' && (window as any).initFlowbite) {
        (window as any).initFlowbite();
      }
    }
}