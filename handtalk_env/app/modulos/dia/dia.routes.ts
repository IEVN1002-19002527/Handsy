import { Routes } from '@angular/router';

const routes: Routes = [
  {
    path: '',
    loadComponent: () => import('./dia').then(m => m.DiaComponent)
  }
];

export default routes;

