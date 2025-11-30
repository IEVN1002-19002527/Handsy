import { Routes } from '@angular/router';

const routes: Routes = [
  {
    path: '',
    loadComponent: () => import('./mes').then(m => m.MesComponent)
  }
];

export default routes;

