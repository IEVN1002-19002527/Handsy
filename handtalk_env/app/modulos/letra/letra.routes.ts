import { Routes } from '@angular/router';

const routes: Routes = [
  {
    path: '',
    loadComponent: () => import('./letra').then(m => m.LetraComponent)
  }
];

export default routes;

