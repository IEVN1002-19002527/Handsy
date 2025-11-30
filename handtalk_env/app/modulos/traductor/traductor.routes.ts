import { Routes } from '@angular/router';

const routes: Routes = [
  {
    path: '',
    loadComponent: () => import('./traductor').then(m => m.TraductorComponent)
  }
];

export default routes;

