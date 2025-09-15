-- AstraFX Supabase bootstrap (run in Supabase SQL editor)

create extension if not exists "pgcrypto";

create table if not exists public.profiles (
  id uuid primary key references auth.users(id) on delete cascade,
  created_at timestamptz default now(),
  display_name text,
  stripe_customer_id text,
  is_pro boolean default false
);
alter table public.profiles enable row level security;
create policy "profiles_select_own" on public.profiles for select using (auth.uid() = id);
create policy "profiles_insert_self" on public.profiles for insert with check (auth.uid() = id);
create policy "profiles_update_own" on public.profiles for update using (auth.uid() = id);

create table if not exists public.journal_entries (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  created_at timestamptz default now(),
  symbol text,
  timeframe text,
  entry jsonb
);
alter table public.journal_entries enable row level security;
create policy "journal_select_own" on public.journal_entries for select using (auth.uid() = user_id);
create policy "journal_insert_own" on public.journal_entries for insert with check (auth.uid() = user_id);
create policy "journal_update_own" on public.journal_entries for update using (auth.uid() = user_id);
create policy "journal_delete_own" on public.journal_entries for delete using (auth.uid() = user_id);
