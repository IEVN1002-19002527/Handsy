import { Routes } from '@angular/router';

export default [
    {
        path: 'letra',
        loadComponent: () => import('./letra/letra').then(m => m.LetraComponent)
    },
    {
        path: 'numero',
        loadComponent: () => import('./numero/numero').then(m => m.NumeroComponent)
    },
    {
        path: 'dia',
        loadComponent: () => import('./dia/dia').then(m => m.DiaComponent)
    },
    {
        path: 'mes',
        loadComponent: () => import('./mes/mes').then(m => m.MesComponent)
    },
    {
        path: 'palabra',
        loadComponent: () => import('./palabra/palabra').then(m => m.PalabraComponent)
    },
    {
        path: 'traductor',
        loadComponent: () => import('./traductor/traductor').then(m => m.TraductorComponent)
    }
] as Routes;