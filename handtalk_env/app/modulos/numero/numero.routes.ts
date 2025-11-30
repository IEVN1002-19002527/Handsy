import { Routes } from '@angular/router';

const routes: Routes = [
  {
    path: '',
    loadComponent: () => import('./numero').then(m => m.NumeroComponent)
  }
];

export default routes;

