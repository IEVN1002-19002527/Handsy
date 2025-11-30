import { Routes } from '@angular/router';

const routes: Routes = [
  {
    path: '',
    loadComponent: () => import('./palabra').then(m => m.PalabraComponent)
  }
];

export default routes;

