import { Routes } from '@angular/router';

export const routes: Routes = [
    {
        path: '',
        redirectTo: '/modulos/letra',
        pathMatch: 'full'
    },
    {
        path: 'modulos',
        loadChildren: () => import('./modulos/modulos.routes').then(m => m.default)
    }
];